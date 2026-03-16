"""
Example script for running dreaming on a dataset.
The idea is that there are ground_truth ("reference") video clips, and we dream the same clips given some initial context.

After dreaming, we have two sets of videos which, barring the intrinsic noise of the game environment (e.g., randomness of other players),
should be identical if model was ideal.
"""

import argparse
from pathlib import Path
import os
import subprocess

import cv2
from tensordict import TensorDict
import torch as th
from tqdm import tqdm
import numpy as np
import ffmpegcv
from PIL import Image

import wham.utils as utils


parser = argparse.ArgumentParser(description="Run dreaming.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--data_path", type=str, required=True, help="Path to the directory that contains the ground truth data to dream for.")
parser.add_argument("--output", type=str, default="dreaming_output", help="Path to the directory where output should be put.")
parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process.")
parser.add_argument("--metadata_config", type=str, default="configs/metadata_custom_tag.config", help="Path to metadata tag config for origin field.")


parser.add_argument(
    "--protocol",
    type=str,
    default="base",
    choices=["base", "comprehensive"],
    help="What protocol to use for the dreaming. base = action conditioned, comprehensive = dream actions as well.",
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dreaming. Higher batch_size uses more VRAM but overall is faster.")
parser.add_argument("--context_length", type=int, default=10, help="Number of frames to use an initial context.")
parser.add_argument("--steps_to_dream", type=int, default=10, help="Batch size for dreaming.")

parser.add_argument("--sampling_temperature", type=float, default=0.9, help="Temperature for sampling from the model.")
parser.add_argument("--sampling_top_k", type=int, default=None, help="Top-k for sampling from the model.")
parser.add_argument("--sampling_top_p", type=float, default=None, help="Top-p for sampling from the model.")


def get_context_data(image_context, action_context, action_sequences):
    # Make sure we have CHW images:
    assert image_context.shape[-3] == 3, "Image context should be CHW"

    image_context = th.from_numpy(image_context).cuda()
    action_data = th.from_numpy(action_context).float().cuda()
    action_sequences = th.from_numpy(action_sequences).float().cuda() if action_sequences is not None else None

    return TensorDict({"images": image_context, "actions_output": action_data}, batch_size=image_context.shape[:2])


def add_video_metadata(file_path, metadata_config):
    # Construct the exiftool command
    cmd = [
        'exiftool',
        '-config', metadata_config,
        f'-ProgramName=\"{utils.PROGRAM_NAME}\"',
        '-overwrite_original',
        file_path
    ]

    try:
        # Execute the exiftool command
        subprocess.run(cmd, check=True)
        print(f"Metadata modified successfully.")
        # Print the new file metadata
        cmd_output = [
            'exiftool',
            file_path
        ]
        subprocess.run(cmd_output, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error modifying metadata: {e}")


@th.no_grad()
def do_dreaming(model, image_context, action_context, args, action_sequences=None):
    """
    image_contect and action_context provide the initial context for the model to dream from.

    If action_sequences (batch_size, args.steps_to_dream, action_dim) is provided, then model will be prompted with these actions.
    """
    context_data = get_context_data(image_context, action_context, action_sequences)
    encoded_context_data = model.encode_context(context_data)

    encoded_action_sequences = None
    if action_sequences is not None:
        assert action_sequences.shape[1] == args.steps_to_dream, "action_sequences should have shape (batch_size, args.steps_to_dream, action_dim)"
        action_sequences = TensorDict({"actions_output": action_sequences}, batch_size=action_sequences.shape[:2]).cuda()
        encoded_action_sequences = model.encode_context(action_sequences)

    encoded_dreamt_steps = []

    for dream_step in range(args.steps_to_dream):
        encoded_predicted_step, _ = model.predictor.predict_next_step(
            encoded_context_data, temperature=args.sampling_temperature, top_k=args.sampling_top_k, top_p=args.sampling_top_p, min_tokens_to_keep=1
        )

        # Remove first step from context if we are at the max context length:
        if encoded_context_data.shape[1] == args.context_length:
            encoded_context_data = encoded_context_data[:, 1:]

        # Add predicted image + action to the context
        append_step = encoded_predicted_step
        if encoded_action_sequences is not None:
            # Replace predicted action with real action
            append_step["actions_output"] = encoded_action_sequences["actions_output"][:, [dream_step], :]
        encoded_context_data = th.cat((encoded_context_data, append_step), dim=1)

        encoded_dreamt_steps.append(encoded_predicted_step)

    # Decode everything
    dreamed_images = []
    actions_during_dream = []
    for seq_i in range(args.steps_to_dream):
        decoded_step = model.decode_context(encoded_dreamt_steps[seq_i])
        dreamed_images.append(decoded_step["images"][:, [0]].cpu().numpy())
        actions_during_dream.append(decoded_step["actions_output"][:, [0]].cpu().numpy())

    dreamed_images = np.concatenate(dreamed_images, axis=1)
    actions_during_dream = np.concatenate(actions_during_dream, axis=1)

    return dreamed_images, actions_during_dream


@th.no_grad()
def encode_decode_images(model, images):
    """
    Pass ground_truth images through the encoding/decoding process of the model.
    """
    context = TensorDict({"images": th.from_numpy(images).cuda()}, batch_size=images.shape[:2])
    output_images = []
    for seq_i in range(images.shape[1]):
        encoded_images = model.encode_context(context[:, [seq_i]])
        decoded_images = model.decode_context(encoded_images)
        output_images.append(decoded_images["images"].cpu().numpy())
    return np.concatenate(output_images, axis=1)


def main(args):
    total_video_length = args.context_length + args.steps_to_dream

    # Now, load the model:
    model_path = Path(args.model_path)
    assert model_path.is_file(), "Could not find the model!"
    model = utils.load_model_from_checkpoint(model_path).cuda()

    # Glob the dataset to find all the ground truth segments we want to construct a dream for:
    data_path = Path(args.data_path)
    ground_truth_files = list(data_path.rglob("*.npz"))
    num_dreams = len(ground_truth_files)

    if args.max_files is not None:
        # Sort to make sure we always get the same files
        ground_truth_files = sorted(ground_truth_files)
        ground_truth_files = ground_truth_files[: args.max_files]
        num_dreams = len(ground_truth_files)

    output_path = Path(args.output)
    os.makedirs(output_path, exist_ok=True)

    print("=" * 100)
    print(f"GENERATING DREAMS OF {num_dreams} SEGMENTS")
    print(f"WRITING TO {args.output}")
    print("=" * 100)

    dreams_created = 0
    with tqdm(total=num_dreams, desc="Dreams") as pbar:
        while ground_truth_files:
            # Load batch_size headers:
            batches = min(args.batch_size, len(ground_truth_files))
            batched_image_context = []
            batched_image_sequence = []
            batched_action_context = []
            batched_action_sequence = []
            episode_names = []
            for i in range(batches):
                episode = ground_truth_files.pop()
                episode_names.append(episode)
                try:
                    data = np.load(episode)
                    images = data["images"]
                    actions = data["actions"]
                except Exception:
                    print(f"Failed to load episode {episode} - skipping.")
                    continue

                if actions.shape[0] < total_video_length:
                    # We want to make sure we have ground_truth comparisons for the entire dream, so we ensure the episode is long enough
                    raise ValueError(f"Episode {episode} is too short to dream from. It has {actions.shape[0]} steps, but we need at least {total_video_length}.")
                batched_image_context.append(images[: args.context_length])
                batched_image_sequence.append(images[args.context_length: total_video_length])
                batched_action_context.append(actions[: args.context_length])
                batched_action_sequence.append(actions[args.context_length: total_video_length])

            image_context = np.array(batched_image_context)
            image_sequences = np.array(batched_image_sequence)
            action_context = np.array(batched_action_context)
            action_sequences = np.array(batched_action_sequence)

            if args.protocol == "comprehensive":
                # We do not need to pass in the action sequences for comprehensive protocol
                action_sequences = None

            full_image_sequence = np.concatenate((image_context, image_sequences), axis=1)

            dreamt_images, actions_during_dream = do_dreaming(model, image_context, action_context, args, action_sequences=action_sequences)
            encoded_decoded_images_batch = encode_decode_images(model, full_image_sequence)

            pbar.update(batches)
            dreams_created += batches

            # Save the dreams:
            # We are aiming to mimic the folder structure of the ground truth dataset, so use the episode names
            # but make them relative to our output folder:
            for i, dream in enumerate(dreamt_images):
                episode = episode_names[i]
                output_file = output_path / episode.relative_to(data_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    output_file,
                    context_length=args.context_length,
                    steps_to_dream=args.steps_to_dream,
                    raw_context=image_context[i],
                    dreamt_images=dream,
                    all_actions=np.concatenate((action_context[i], actions_during_dream[i])),
                    encoded_decoded_ground_truth_images=encoded_decoded_images_batch[i],
                )

                video_file = str(output_file.with_suffix(".mp4"))
                writer = ffmpegcv.VideoWriter(video_file, None, utils.DREAMING_FPS)
                full_sequence = np.concatenate((image_context[i], dream), axis=0)
                for frame in full_sequence:
                    img = frame.transpose(1, 2, 0).astype(np.uint8).copy()
                    # Please DO NOT remove this watermark. This will infringe upon the repo's license agreement
                    (text_width, _), _ = cv2.getTextSize(utils.WATERMARK_TEXT, utils.WATERMARK_FONT, utils.WATERMARK_FONT_SCALE, utils.WATERMARK_FONT_THICKNESS)
                    x = img.shape[1] - text_width - 10  # 10 pixels from the right edge
                    y = img.shape[0] - 10  # 10 pixels from the bottom edge
                    cv2.putText(img, utils.WATERMARK_TEXT, (x, y), utils.WATERMARK_FONT, utils.WATERMARK_FONT_SCALE, utils.WATERMARK_FONT_COLOR, utils.WATERMARK_FONT_THICKNESS)

                    # Add image metadata
                    pil_image = Image.fromarray(img) 
                    pil_image.info['Id'] = 0x0131
                    pil_image.info['Type'] = 2
                    pil_image.info['Value'] = utils.PROGRAM_NAME.encode("utf-8")
                    pil_image.info['Len'] = len(utils.PROGRAM_NAME) + 1

                    # Convert pil_image to a CV2 format for the video writer
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    writer.write(cv_image)
                writer.release()
                add_video_metadata(video_file, args.metadata_config)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
