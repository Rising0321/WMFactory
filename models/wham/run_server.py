import argparse
from dataclasses import dataclass, field
import json
import copy
import multiprocessing as mp
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
import io
import zipfile
import queue
import time
import random
import logging

from tensordict import TensorDict
import cv2
from flask import Flask, request, make_response, send_file
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch as th

from wham.utils import load_model_from_checkpoint, POS_BINS_BOUNDARIES, POS_BINS_MIDDLE

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Simple Dreamer")
parser.add_argument("--model", type=str, required=True, help="Path to the model file for the local runs")
parser.add_argument("--debug", action="store_true", help="Enable flask debug mode.")
parser.add_argument("--random_model", action="store_true", help="Use randomly initialized model instead of the provided one")
parser.add_argument("--port", type=int, default=5000)

parser.add_argument("--max_concurrent_jobs", type=int, default=30, help="Maximum number of jobs that can be run concurrently on this server.")
parser.add_argument("--max_dream_steps_per_job", type=int, default=10, help="Maximum number of dream steps each job can request.")
parser.add_argument("--max_job_lifespan", type=int, default=60 * 10, help="Maximum number of seconds we keep run around if not polled.")

parser.add_argument("--image_width", type=int, default=300, help="Width of the image")
parser.add_argument("--image_height", type=int, default=180, help="Height of the image")

parser.add_argument("--max_batch_size", type=int, default=3, help="Maximum batch size for the dreamer workers")

PREDICTION_JSON_FILENAME = "predictions.json"
# Minimum time between times we check when to delete jobs. We do this when adding new jobs.
JOB_CLEANUP_CHECK_RATE = timedelta(seconds=10)

MAX_CANCELLED_ID_QUEUE_SIZE = 100

DEFAULT_SAMPLING_SETTINGS = {
    "temperature": 0.9,
    "top_k": None,
    "top_p": 1.0,
    "max_context_length": 10,
}


def float_or_none(string):
    if string.lower() == "none":
        return None
    return float(string)


def be_image_preprocess(image, target_width, target_height):
    # If target_width and target_height are specified, resize the image.
    if target_width is not None and target_height is not None:
        # Make sure we do not try to resize if the image is already the correct size.
        if image.shape[1] != target_width or image.shape[0] != target_height:
            image = cv2.resize(image, (target_width, target_height))
    return np.transpose(image, (2, 0, 1))


def action_vector_to_be_action_vector(action):
    # Preprocess a BE action vector from 16 numbers with:
    #  12 buttons [0, 1] and 4 stick directions [-1, 1]
    # to discrete actions valid for the token model
    #  12 buttons [0, 1] and 4 stick directions {discrete bin}
    action[-4:] = np.digitize(action[-4:], bins=POS_BINS_BOUNDARIES) - 1
    return action


def be_action_vector_to_action_vector(action):
    # Preprocess a BE action vector into unified space
    for stick_index in range(-4, 0):
        action[stick_index] = POS_BINS_MIDDLE[int(action[stick_index])]
    return action



@dataclass
class DreamJob:
    job_id: str
    sampling_settings: dict
    num_predictions_remaining: int
    num_predictions_done: int
    # (B, T, C, H, W)
    context_images: th.Tensor
    context_actions: th.Tensor
    # Tokens that will replace the context_images if they are provided
    context_tokens: list
    # This will replace the dreamed action if provided.
    # For every step, we remove the first action until exhausted
    actions_to_take: th.Tensor = None


@dataclass
class DreamJobResult:
    job_id: str
    dream_step_index: int
    # (B, 1, C, H, W)
    dreamt_image: th.Tensor
    dreamt_action: th.Tensor
    dreamt_tokens: th.Tensor
    result_creation_time: datetime = field(default_factory=datetime.now)



def setup_and_load_model_be_model(args):
    model = load_model_from_checkpoint(args.model)
    th.set_float32_matmul_precision("high")
    th.backends.cuda.matmul.allow_tf32 = True
    return model


def get_job_batchable_information(job):
    """Return comparable object of job information. Used for batching"""
    context_length = job.context_images.shape[1]
    return (context_length, job.sampling_settings)


def fetch_list_of_batchable_jobs(job_queue, cancelled_ids_set, max_batch_size, timeout=1):
    """Return a list of jobs (or empty list) that can be batched together"""
    batchable_jobs = []
    required_job_info = None
    while len(batchable_jobs) < max_batch_size:
        try:
            job = job_queue.get(timeout=timeout)
        except queue.Empty:
            break
        # If pipe breaks, also gracefully return
        except OSError:
            break
        if job.job_id in cancelled_ids_set:
            # This job was cancelled, so discard it completely
            continue
        job_info = get_job_batchable_information(job)
        if required_job_info is None:
            required_job_info = job_info
        elif required_job_info != job_info:
            # This job is not batchable, put it back
            job_queue.put(job)
            # we assume here that, generally, the others jobs would also be
            # invalid. So we just return the batchable jobs we have instead
            # of going through more.
            break
        batchable_jobs.append(job)
    return batchable_jobs


def update_cancelled_jobs(cancelled_ids_queue, cancelled_ids_deque, cancelled_ids_set):
    """IN-PLACE Update cancelled_ids_set with new ids from the queue"""
    has_changed = False
    while not cancelled_ids_queue.empty():
        try:
            cancelled_id = cancelled_ids_queue.get_nowait()
        except queue.Empty:
            break
        cancelled_ids_deque.append(cancelled_id)
        has_changed = True

    if has_changed:
        cancelled_ids_set.clear()
        cancelled_ids_set.update(cancelled_ids_deque)


def predict_step(context_data, sampling_settings, model, tokens=None):
    with th.no_grad():
        predicted_step = model.predict_next_step(context_data, min_tokens_to_keep=1, tokens=tokens, **sampling_settings)
    return predicted_step


def dreamer_worker(job_queue, result_queue, cancelled_jobs_queue, quit_flag, device_to_use, args):
    logger = logging.getLogger(f"dreamer_worker {device_to_use}")
    logger.info("Loading up model...")
    model = setup_and_load_model_be_model(args)
    model = model.to(device_to_use)
    logger.info("Model loaded. Fetching results")

    cancelled_ids_deque = deque(maxlen=MAX_CANCELLED_ID_QUEUE_SIZE)
    cancelled_ids_set = set()

    while not quit_flag.is_set():
        update_cancelled_jobs(cancelled_jobs_queue, cancelled_ids_deque, cancelled_ids_set)
        batchable_jobs = fetch_list_of_batchable_jobs(job_queue, cancelled_ids_set, max_batch_size=args.max_batch_size)
        if len(batchable_jobs) == 0:
            continue
        sampling_settings = batchable_jobs[0].sampling_settings
        # make better way for passing these arguments around. sampling_settings
        # is passed as kwargs to predicting step, but max_context_length is not part of valid
        # keys there, so we need to pop it out.
        max_context_length = sampling_settings.pop("max_context_length")

        images = [job.context_images[:, :max_context_length] for job in batchable_jobs]
        actions = [job.context_actions[:, :max_context_length] for job in batchable_jobs]
        tokens = [job.context_tokens for job in batchable_jobs]

        images = th.concat(images, dim=0).to(device_to_use)
        actions = th.concat(actions, dim=0).to(device_to_use)

        context_data = TensorDict({
            "images": images,
            "actions_output": actions
        }, batch_size=images.shape[:2])

        predicted_step, predicted_image_tokens = predict_step(context_data, sampling_settings, model, tokens)

        predicted_step = predicted_step.cpu()
        predicted_images = predicted_step["images"]
        predicted_actions = predicted_step["actions_output"]
        predicted_image_tokens = predicted_image_tokens.cpu()

        for job_i, job in enumerate(batchable_jobs):
            image_context = job.context_images
            action_context = job.context_actions
            token_context = job.context_tokens
            # Keep batch dimension
            dreamt_image = predicted_images[job_i].unsqueeze(0)
            dreamt_action = predicted_actions[job_i].unsqueeze(0)
            dreamt_tokens = predicted_image_tokens[job_i].unsqueeze(0)

            # Replace the dreamed action if provided
            actions_to_take = job.actions_to_take
            if actions_to_take is not None and actions_to_take.shape[1] > 0:
                dreamt_action = actions_to_take[:, 0:1]
                # Remove the action we took
                actions_to_take = actions_to_take[:, 1:]
                if actions_to_take.shape[1] == 0:
                    actions_to_take = None

            result_queue.put(DreamJobResult(
                job_id=job.job_id,
                dream_step_index=job.num_predictions_done,
                dreamt_image=dreamt_image,
                dreamt_action=dreamt_action,
                dreamt_tokens=dreamt_tokens
            ))

            # Add job back in the queue if we have more steps to do
            if job.num_predictions_remaining > 0:
                # Stack the dreamt image and action to the context
                if image_context.shape[1] >= max_context_length:
                    image_context = image_context[:, 1:]
                    action_context = action_context[:, 1:]
                    token_context = token_context[1:]
                image_context = th.cat([image_context, dreamt_image], dim=1)
                action_context = th.cat([action_context, dreamt_action], dim=1)
                token_context.append(dreamt_tokens[0, 0].tolist())
                # We need to add context length back to sampling settings...
                # add some better way of passing these settings around
                job.sampling_settings["max_context_length"] = max_context_length
                job_queue.put(DreamJob(
                    job_id=job.job_id,
                    sampling_settings=job.sampling_settings,
                    num_predictions_remaining=job.num_predictions_remaining - 1,
                    num_predictions_done=job.num_predictions_done + 1,
                    context_images=image_context,
                    context_actions=action_context,
                    context_tokens=token_context,
                    actions_to_take=actions_to_take
                ))


class DreamerServer:
    def __init__(self, num_workers, args):
        self.num_workers = num_workers
        self.args = args
        self.model = None
        self.jobs = mp.Queue(maxsize=args.max_concurrent_jobs)
        self.results_queue = mp.Queue()
        self.cancelled_jobs = set()
        self.cancelled_jobs_queues = [mp.Queue() for _ in range(num_workers)]
        # job_id -> results
        self._last_result_cleanup = datetime.now()
        self._max_job_lifespan_datetime = timedelta(seconds=args.max_job_lifespan)
        self.local_results = defaultdict(list)
        self.logger = logging.getLogger("DreamerServer")

    def get_details(self):
        details = {
            "model_file": self.args.model,
            "max_concurrent_jobs": self.args.max_concurrent_jobs,
            "max_dream_steps_per_job": self.args.max_dream_steps_per_job,
            "max_job_lifespan": self.args.max_job_lifespan,
        }
        return json.dumps(details)

    def _check_if_should_remove_old_jobs(self):
        time_now = datetime.now()
        # Only cleanup every JOB_CLEANUP_CHECK_RATE seconds at most
        if time_now - self._last_result_cleanup < JOB_CLEANUP_CHECK_RATE:
            return

        self._last_result_cleanup = time_now
        # First add existing results to the local results
        self._gather_new_results()
        # Check if we should remove old jobs
        job_ids = list(self.local_results.keys())
        for job_id in job_ids:
            results = self.local_results[job_id]
            # If newest result is older than max_job_lifespan, remove the job
            if time_now - results[-1].result_creation_time > self._max_job_lifespan_datetime:
                self.logger.info(f"Deleted job {job_id} because it was too old. Last result was {results[-1].result_creation_time}")
                del self.local_results[job_id]

    def add_new_job(self, request, request_json):
        """
        Add new dreaming job to the queues.
        Request should have:


        Returns: json object with new job id
        """
        self._check_if_should_remove_old_jobs()

        sampling_settings = copy.deepcopy(DEFAULT_SAMPLING_SETTINGS)
        if "num_steps_to_predict" not in request_json:
            return make_response("num_steps_to_predict not in request", 400)
        num_steps_to_predict = request_json['num_steps_to_predict']
        if num_steps_to_predict > self.args.max_dream_steps_per_job:
            return make_response(f"num_steps_to_predict too large. Max {self.args.max_dream_steps_per_job}", 400)

        num_parallel_predictions = int(request_json['num_parallel_predictions']) if 'num_parallel_predictions' in request_json else 1

        if (self.jobs.qsize() + num_parallel_predictions) >= self.args.max_concurrent_jobs:
            return make_response(f"Too many jobs already running. Max {self.args.max_concurrent_jobs}", 400)

        for key in sampling_settings:
            sampling_settings[key] = float_or_none(request_json[key]) if key in request_json else sampling_settings[key]

        context_images = []
        context_actions = []
        context_tokens = []
        future_actions = []

        for step in request_json["steps"]:
            image_path = step["image_name"]
            image = np.array(Image.open(request.files[image_path].stream))
            image = be_image_preprocess(image, target_width=self.args.image_width, target_height=self.args.image_height)
            context_images.append(th.from_numpy(image))

            action = step["action"]
            action = action_vector_to_be_action_vector(action)
            context_actions.append(th.tensor(action))

            tokens = step["tokens"]
            context_tokens.append(tokens)

        future_actions = None
        if "future_actions" in request_json:
            future_actions = []
            for step in request_json["future_actions"]:
                # The rest is the action vector
                action = step["action"]
                action = action_vector_to_be_action_vector(action)
                # Add sequence and batch dimensions
                future_actions.append(th.tensor(action))

        # Add batch dimensions
        context_images = th.stack(context_images).unsqueeze(0)
        context_actions = th.stack(context_actions).unsqueeze(0)
        future_actions = th.stack(future_actions).unsqueeze(0) if future_actions is not None else None

        list_of_job_ids = []
        for _ in range(num_parallel_predictions):
            job_id = uuid.uuid4().hex
            self.jobs.put(DreamJob(
                job_id=job_id,
                sampling_settings=sampling_settings,
                num_predictions_remaining=num_steps_to_predict,
                num_predictions_done=0,
                context_images=context_images,
                context_actions=context_actions,
                context_tokens=context_tokens,
                actions_to_take=future_actions
            ))
            list_of_job_ids.append(job_id)

        job_queue_size = self.jobs.qsize()
        return json.dumps({"job_ids": list_of_job_ids, "current_jobs_in_queue": job_queue_size})

    def _gather_new_results(self):
        if not self.results_queue.empty():
            for _ in range(self.results_queue.qsize()):
                result = self.results_queue.get()
                if result.job_id in self.cancelled_jobs:
                    # Discard result if job was cancelled
                    continue
                self.local_results[result.job_id].append(result)

    def get_new_results(self, request, request_json):
        if "job_ids" not in request_json:
            return make_response("job_ids not in request", 400)
        self._gather_new_results()
        job_ids = request_json["job_ids"]
        if not isinstance(job_ids, list):
            job_ids = [job_ids]
        return_results = []
        for job_id in job_ids:
            if job_id in self.local_results:
                return_results.append(self.local_results[job_id])
                del self.local_results[job_id]

        if len(return_results) == 0:
            return make_response("No new responses", 204)

        output_json = []
        output_image_bytes = {}
        for job_results in return_results:
            for result in job_results:
                action = result.dreamt_action.numpy()
                # Remember to remove batch and sequence dimensions
                action = be_action_vector_to_action_vector(action[0, 0].tolist())
                dreamt_tokens = result.dreamt_tokens[0, 0].tolist()
                image_filename = f"{result.job_id}_{result.dream_step_index}.png"
                output_json.append({
                    "job_id": result.job_id,
                    "dream_step_index": result.dream_step_index,
                    "action": action,
                    "tokens": dreamt_tokens,
                    "image_filename": image_filename
                })

                image_bytes = io.BytesIO()
                # this probably is not as smooth as it could be
                T.ToPILImage()(result.dreamt_image[0, 0]).save(image_bytes, format="PNG")
                output_image_bytes[image_filename] = image_bytes.getvalue()

        # Write a zip file with all the images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, "w") as z:
            for filename, bytes in output_image_bytes.items():
                z.writestr(filename, bytes)
            # Write the json
            z.writestr(PREDICTION_JSON_FILENAME, json.dumps(output_json))

        zip_bytes.seek(0)

        return send_file(
            zip_bytes,
            mimetype="zip",
            as_attachment=True,
            download_name=f"dreaming_results_{timestamp}.zip"
        )

    def cancel_job(self, request, request_json):
        if "job_id" not in request_json:
            return make_response("job_id not in request", 400)
        job_id = request_json["job_id"]
        self.cancelled_jobs.add(job_id)
        # Cancel all jobs in the queue with this id
        for job_queue in self.cancelled_jobs_queues:
            job_queue.put(job_id)
        return make_response("OK", 200)


def main_run(args):
    app = Flask(__name__)

    num_workers = th.cuda.device_count()
    if num_workers == 0:
        raise RuntimeError("No CUDA devices found. Cannot run Dreamer.")

    server = DreamerServer(num_workers, args)
    quit_flag = mp.Event()

    # Start the dreamer worker(s)
    dreamer_worker_processes = []
    for device_i in range(num_workers):
        device = f"cuda:{device_i}"
        dreamer_worker_process = mp.Process(
            target=dreamer_worker,
            args=(server.jobs, server.results_queue, server.cancelled_jobs_queues[device_i], quit_flag, device, args)
        )
        dreamer_worker_process.daemon = True
        dreamer_worker_process.start()
        dreamer_worker_processes.append(dreamer_worker_process)

    # Add the API endpoints
    @app.route('/')
    def details():
        return server.get_details()

    @app.route('/new_job', methods=['POST'])
    def new_job():
        request_json = json.loads(request.form["json"])
        return server.add_new_job(request, request_json)

    @app.route('/get_job_results', methods=['GET'])
    def get_results():
        # the "Json" is now in regular GET payload/parameters
        request_json = {"job_ids": request.args.getlist("job_ids")}
        return server.get_new_results(request, request_json)

    @app.route('/cancel_job', methods=['GET'])
    def cancel_job():
        request_json = request.args.to_dict()
        return server.cancel_job(request, request_json)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)

    # Cleanup
    quit_flag.set()
    for dreamer_worker_process in dreamer_worker_processes:
        dreamer_worker_process.join()


if __name__ == '__main__':
    args = parser.parse_args()
    main_run(args)
