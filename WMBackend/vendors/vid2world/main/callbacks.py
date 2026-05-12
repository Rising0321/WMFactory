import os
import time
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from utils.save_video import log_local, prepare_to_log, tensor_to_mp4
from utils.metrics import Evaluator
import numpy as np
import scipy

from pytorch_fid.fid_score import calculate_frechet_distance

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images=8, clamp=True, rescale=True, save_dir=None, \
                to_local=False, log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.to_local = to_local
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        if self.to_local:
            ## default save dir
            self.save_dir = os.path.join(save_dir, "images")
            os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "val"), exist_ok=True)

    def log_to_tensorboard(self, pl_module, batch_logs, filename, split, save_fps=3): # rt-1 should be fps=3
        """ log images and videos to tensorboard """        
        global_step = pl_module.global_step
        # dict_keys(['image_condition', 'reconst', 'condition', 'samples'])
        
        for key in batch_logs:
            value = batch_logs[key]
            tag = "gs%d-%s/%s-%s"%(global_step, split, filename, key)
            if isinstance(value, list) and isinstance(value[0], str):
                captions = ' |------| '.join(value)
                pl_module.logger.experiment.add_text(tag, captions, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 5:
                video = value
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, n*h, 1*w]
                grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                pl_module.logger.experiment.add_video(tag, grid, fps=save_fps, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                img = value
                grid = torchvision.utils.make_grid(img, nrow=int(n), padding=0)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                pl_module.logger.experiment.add_image(tag, grid, global_step=global_step)
            else:
                pass

    @rank_zero_only
    def log_batch_imgs(self, pl_module, batch, batch_idx, split="train"):
        """ generate images, then save and log to tensorboard """
        skip_freq = self.batch_freq if split == "train" else 5000
        # Reminder: if the training log does not contain logged videos, it might be due to:
        # the local batch size is smaller than skip_freq
        if (batch_idx+1) % skip_freq == 0 or split == "val": # log all videos in validation set
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                log_func = pl_module.log_images
                # Extract ar parameter to avoid duplicate keyword argument
                log_kwargs = self.log_images_kwargs.copy()
                ar_value = log_kwargs.pop("ar", None)  # Remove 'ar' from kwargs if exists
                
                if ar_value is False or ar_value is None:
                    # Non-autoregressive mode or ar not specified
                    batch_logs = log_func(batch, split=split, sampled_img_num=batch['video'].shape[0], **log_kwargs)
                else:
                    # Autoregressive mode (ar=True)
                    batch_logs = log_func(batch, split=split, sampled_img_num=batch['video'].shape[0], ar=True, ar_noise_schedule=1, **log_kwargs)
            
            ## process: move to CPU and clamp
            batch_logs = prepare_to_log(batch_logs, self.max_images, self.clamp)
            torch.cuda.empty_cache()
            
            filename = "ep{}_idx{}_rank{}".format(
                pl_module.current_epoch,
                batch_idx,
                pl_module.global_rank)
            if self.to_local:
                mainlogger.info("Log [%s] batch <%s> to local ..."%(split, filename))
                filename = "gs{}_".format(pl_module.global_step) + filename
                log_local(batch_logs, os.path.join(self.save_dir, split), filename, save_fps=3)
            else:
                mainlogger.info("Log [%s] batch <%s> to tensorboard ..."%(split, filename))
                batch_logs['video'] = batch['video']
                self.log_to_tensorboard(pl_module, batch_logs, filename, split, save_fps=3)
            mainlogger.info('Finish!')

            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.batch_freq != -1:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if self.batch_freq != -1:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if int((pl.__version__).split('.')[1])>=7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

class MetricsLogger(Callback):
    def __init__(self, env=None, log_every_n_train_steps=1000, buffer_size=32, i3d_model_path=None, max_batchsize=2, log_images_kwargs=None, save_dir=None, evaluate=True, skip_batch_if_all_exist=True):
        """
        evaluator: see utils.metrics.py
        log_every_n_train_steps: log every n train steps
        buffer_size: the number of batches to accumulate for calculating metrics, to avoid large fluctuations, see below
        """
        super().__init__()
        self.env = env
        self.log_every_n_train_steps = log_every_n_train_steps
        self.buffer_size = buffer_size
        self.i3d_model_path = i3d_model_path
        self.max_batchsize = max_batchsize
        self.save_dir = save_dir
        self.evaluate = evaluate
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}   
        self.skip_batch_if_all_exist = skip_batch_if_all_exist
        self.evaluator = Evaluator(
            i3d_model_path=self.i3d_model_path,
            max_batchsize=self.max_batchsize,
            device='cuda:0',
            env=self.env,
            save_dir=self.save_dir,
            init_eval_models=self.evaluate
        )
        self.val_metrics_buffer = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            if self.skip_batch_if_all_exist:
                all_exist = True
                for path in batch['path']:
                    npz_file_name = os.path.splitext(os.path.basename(path))[0]
                    save_folder = os.path.join(self.save_dir, npz_file_name)
                    pred_path = os.path.join(save_folder, 'pred_video.mp4')
                    gt_path = os.path.join(save_folder, 'gt_video.mp4')
                    if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
                        all_exist = False
                        break
                    else:
                        print(f"Already exists: pred_path: {pred_path}, gt_path: {gt_path}")
                if all_exist:
                    print(f"All videos already exist, skip this batch")
                    return
            x_pred, x_reconst = self.get_predictions(pl_module, batch, split="val")
            x_pred=x_pred.clamp(-1, 1)
            x_gt = batch['video']
            metrics = self.evaluator.evaluate_all(x_pred, x_gt, raw=True, path_dict=batch['path'], evaluate=self.evaluate)
            # dict_keys(['mse', 'psnr', 'ssim', 'lpips', 'raw_gt_features', 'raw_pred_features', 'real_stats', 'fake_stats'])
            if metrics is not None:
                for key, value in metrics.items():
                    if key in ["raw_gt_features", "raw_pred_features", "real_stats", "fake_stats"]:
                        # value is a list of numpy arrays
                        tensorized_value = torch.tensor(value, device=pl_module.device)
                        if key not in self.val_metrics_buffer.keys():
                            self.val_metrics_buffer[key] = tensorized_value
                        else:
                            self.val_metrics_buffer[key]= torch.cat([self.val_metrics_buffer[key], tensorized_value], dim=0)
                    else:
                        if key not in self.val_metrics_buffer.keys():
                            self.val_metrics_buffer[key] = torch.tensor(value, device=pl_module.device).unsqueeze(0)
                        else:
                            self.val_metrics_buffer[key]= torch.cat([self.val_metrics_buffer[key], torch.tensor(value, device=pl_module.device).unsqueeze(0)], dim=0)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Calculate metrics for each batch during validation and average them at the end
        """
        if not self.evaluate:
            return
        gather_val_metrics_buffer = pl_module.all_gather(self.val_metrics_buffer)
        for key, value in gather_val_metrics_buffer.items():
            gather_val_metrics_buffer[key] = value.reshape(-1, *value.shape[2:])
        with torch.no_grad():
            if pl_module.global_rank == 0:
                for key, value in gather_val_metrics_buffer.items():
                    if key == "raw_gt_features":
                        raw_gt_features = value.cpu().numpy()
                        mu_true = np.mean(raw_gt_features, axis=0)
                        sigma_true = np.cov(raw_gt_features, rowvar=False)
                    elif key == "raw_pred_features":
                        raw_pred_features = value.cpu().numpy()
                        mu_pred = np.mean(raw_pred_features, axis=0)
                        sigma_pred = np.cov(raw_pred_features, rowvar=False)
                    elif key == "real_stats":
                        real_stats = value.cpu().numpy()
                        mu_real = np.mean(real_stats, axis=0)
                        sigma_real = np.cov(real_stats, rowvar=False)
                    elif key == "fake_stats":
                        fake_stats = value.cpu().numpy()
                        mu_fake = np.mean(fake_stats, axis=0)
                        sigma_fake = np.cov(fake_stats, rowvar=False)
                    else:
                        mean_value = torch.mean(value)
                        pl_module.logger.experiment.add_scalar(f"val/{key}", mean_value, global_step=pl_module.global_step)
                        mainlogger.info(f"Epoch end val {key}: {mean_value}")
                m = np.square(mu_pred - mu_true).sum()
                s, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
                fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))
                fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
                pl_module.logger.experiment.add_scalar(f"val/fid", fid, global_step=pl_module.global_step)
                pl_module.logger.experiment.add_scalar(f"val/fvd", fvd, global_step=pl_module.global_step)
                mainlogger.info(f"Epoch end val fvd: {fvd}")
                mainlogger.info(f"Epoch end val fid: {fid}")
            # Clear buffer and broadcast to all processes
            self.val_metrics_buffer = {}
            # Clear GPU cache after validation
            torch.cuda.empty_cache()

    def get_predictions(self, pl_module, batch, pred=None, split="train"):
        """
        run the model to get the predictions
        """
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            log_func = pl_module.log_images
            # Extract ar parameter to avoid duplicate keyword argument
            log_kwargs = self.log_images_kwargs.copy()
            ar_value = log_kwargs.pop("ar", None)  # Remove 'ar' from kwargs if exists
            
            if ar_value is False:
                # Non-autoregressive mode
                batch_logs = log_func(batch, split=split, sampled_img_num=batch['video'].shape[0], ar=False, **log_kwargs)
            else:
                # Autoregressive mode (ar=True or ar not specified)
                batch_logs = log_func(batch, split=split, sampled_img_num=batch['video'].shape[0], ar=True, ar_noise_schedule=1, **log_kwargs)        
        for key in batch_logs:
            # dict_keys(['image_condition', 'reconst', 'condition', 'samples'])
            value = batch_logs[key]
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                ## save video grids
                video = value # b,c,t,h,w
                ## only save grayscale or rgb mode
                if video.shape[1] != 1 and video.shape[1] != 3:
                    continue
                if key == "samples":
                    x_pred = video
                elif key == "reconst":
                    x_gt = video
        if is_train:
            pl_module.train()
        return x_pred, x_gt
    
class RECONVIDMetricsLogger(Callback):
    def __init__(self, save_dir=None, batch_frequency=None, log_images_kwargs=None):
        """
        Log images for RECON video, used for future evaluation
        """
        super().__init__()
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}   
        self.save_dir = os.path.join(save_dir, "images")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        import re
        from utils.save_video import tensor_to_mp4

        # Get batch data first
        paths = batch['path']  # list of file paths or names
        curr_time = batch['curr_time']
        
        # --- vvv MODIFICATION START vvv ---
        # Check if all output files for the entire batch already exist
        # Now we need to include time information in the filename to avoid conflicts
        all_files_exist_for_batch = True
        
        for i, path in enumerate(paths):            
            base_name = os.path.splitext(os.path.basename(path))[0]
            sanitized_base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
            # Include time information in filename to make it unique
            time_suffix = f"_t{curr_time[i]:04d}"
            
            # Define all expected output file paths with time information
            expected_files = [
                os.path.join(self.save_dir, f"{sanitized_base_name}{time_suffix}_x_pred.mp4"),
                os.path.join(self.save_dir, f"{sanitized_base_name}{time_suffix}_x_gt.mp4"),
                os.path.join(self.save_dir, f"{sanitized_base_name}{time_suffix}_xgt_first.png"),
                os.path.join(self.save_dir, f"{sanitized_base_name}{time_suffix}_xgt_last.png"),
                os.path.join(self.save_dir, f"{sanitized_base_name}{time_suffix}_xpred_last.png")
            ]
            
            # Check if any file is missing for the current item
            if not all(os.path.exists(f) for f in expected_files):
                all_files_exist_for_batch = False
                break  # If one is missing, no need to check further, process the batch

        if all_files_exist_for_batch:
            mainlogger.info(f"All videos and frames for batch_idx {batch_idx} already exist. Skipping.")
            return # Skip this batch
        # --- ^^^ MODIFICATION END ^^^ ---
        
        with torch.no_grad():
            x_pred, x_reconst = self.get_predictions(pl_module, batch, split="val")
            x_gt = batch['video']
        # Ensure output directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        def norm(t):
            t = t.detach().cpu()
            t = t.clamp(-1, 1)
            return (t + 1.0) / 2.0
        x_pred = norm(x_pred)
        x_gt = norm(x_gt)
        b = x_pred.shape[0]
        
        for i in range(b):
            
            # Sanitize file name and include time information
            base_name = os.path.splitext(os.path.basename(paths[i]))[0]
            base_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
            time_suffix = f"_t{curr_time[i]:04d}"
            
            # Save video: x_pred and x_gt with time information
            tensor_to_mp4(x_pred[i].unsqueeze(0), os.path.join(self.save_dir, f"{base_name}{time_suffix}_x_pred.mp4"), fps=4, rescale=False)
            tensor_to_mp4(x_gt[i].unsqueeze(0), os.path.join(self.save_dir, f"{base_name}{time_suffix}_x_gt.mp4"), fps=4, rescale=False)
            # Save images: first/last frame of x_gt, last frame of x_pred with time information
            torchvision.utils.save_image(x_gt[i,:,0], os.path.join(self.save_dir, f"{base_name}{time_suffix}_xgt_first.png"))
            torchvision.utils.save_image(x_gt[i,:, -1], os.path.join(self.save_dir, f"{base_name}{time_suffix}_xgt_last.png"))
            torchvision.utils.save_image(x_pred[i,:, -1], os.path.join(self.save_dir, f"{base_name}{time_suffix}_xpred_last.png"))
        torch.cuda.empty_cache()

    def get_predictions(self, pl_module, batch, pred=None, split="train"):
        """
        run the model to get the predictions
        """
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            log_func = pl_module.log_images
            batch_logs = log_func(batch, split=split, sampled_img_num=batch['video'].shape[0], ar=True, ar_noise_schedule=1, cond_frame=4, **self.log_images_kwargs)        
        for key in batch_logs:
            # dict_keys(['image_condition', 'reconst', 'condition', 'samples'])
            value = batch_logs[key]
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                ## save video grids
                video = value # b,c,t,h,w
                ## only save grayscale or rgb mode
                if video.shape[1] != 1 and video.shape[1] != 3:
                    continue
                if key == "samples":
                    x_pred = video
                elif key == "reconst":
                    x_gt = video
        if is_train:
            pl_module.train()
        return x_pred, x_gt
    
class CSGOVideoLogger(Callback):
    def __init__(self, env=None, log_every_n_train_steps=1000, buffer_size=32, i3d_model_path=None, max_batchsize=2, log_images_kwargs=None, save_dir=None):
        """
        Log images for CSGO video, used for future evaluation
        """
        super().__init__()
        self.env = env
        self.log_every_n_train_steps = log_every_n_train_steps
        self.buffer_size = buffer_size
        self.i3d_model_path = i3d_model_path
        self.max_batchsize = max_batchsize
        self.save_dir = save_dir
        self.evaluate = False
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}   
        self.evaluator = Evaluator(
            i3d_model_path=self.i3d_model_path,
            max_batchsize=self.max_batchsize,
            device='cuda:0',
            env=self.env,
            save_dir=self.save_dir,
            init_eval_models=self.evaluate
        )
        self.val_metrics_buffer = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # If all corresponding pred_video.mp4 and gt_video.mp4 exist, skip this batch
        all_exist = True
        for path in batch['path']:
            npz_file_name = os.path.splitext(os.path.basename(path))[0]
            save_folder = os.path.join(self.save_dir, npz_file_name)
            pred_path = os.path.join(save_folder, 'pred_video.mp4')
            gt_path = os.path.join(save_folder, 'gt_video.mp4')
            if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
                all_exist = False
                break
            else:
                print(f"Already exists: pred_path: {pred_path}, gt_path: {gt_path}")
        if all_exist:
            print(f"All videos already exist, skip this batch")
            return
        with torch.no_grad():
            x_pred, x_reconst = self.get_predictions(pl_module, batch, split="val")
            x_pred=x_pred.clamp(-1, 1)
            x_gt = batch['video']
            
            # Save actions for each sample in the batch
            actions = batch['action']  # [B, T, 51]
            for idx, path in enumerate(batch['path']):
                npz_file_name = os.path.splitext(os.path.basename(path))[0]
                save_folder = os.path.join(self.save_dir, npz_file_name)
                os.makedirs(save_folder, exist_ok=True)
                # Save action as .npy file
                action_path = os.path.join(save_folder, 'action.npy')
                np.save(action_path, actions[idx].cpu().numpy())
                print(f"Saved action to: {action_path}")
            
            metrics = self.evaluator.evaluate_all(x_pred, x_gt, raw=True, path_dict=batch['path'], evaluate=self.evaluate)
            # dict_keys(['mse', 'psnr', 'ssim', 'lpips', 'raw_gt_features', 'raw_pred_features', 'real_stats', 'fake_stats'])
            assert metrics is None

    def get_predictions(self, pl_module, batch, pred=None, split="train"):
        """
        run the model to get the predictions
        For the csgo evaluation setup, we use 4 condition frames, and auto-regressive generation
        """
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            log_func = pl_module.log_images
            batch_logs = log_func(batch, split=split, sampled_img_num=batch['video'].shape[0], ar=True, ar_noise_schedule=1, cond_frame=4, **self.log_images_kwargs)        
        for key in batch_logs:
            # dict_keys(['image_condition', 'reconst', 'condition', 'samples'])
            value = batch_logs[key]
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                ## save video grids
                video = value # b,c,t,h,w
                ## only save grayscale or rgb mode
                if video.shape[1] != 1 and video.shape[1] != 3:
                    continue
                if key == "samples":
                    x_pred = video
                elif key == "reconst":
                    x_gt = video
        if is_train:
            pl_module.train()
        return x_pred, x_gt

class TrainingMetricsLogger(Callback):
    def __init__(self, log_every_n_train_steps=1000, buffer_size=32, i3d_model_path=None, max_batchsize=2, log_images_kwargs=None):
        """
        evaluator: see utils.metrics.py
        log_every_n_train_steps: log every n train steps
        buffer_size: the number of batches to accumulate for calculating metrics, to avoid large fluctuations, see below
        """
        super().__init__()
        self.log_every_n_train_steps = log_every_n_train_steps
        self.buffer_size = buffer_size
        self.i3d_model_path = i3d_model_path
        self.max_batchsize = max_batchsize
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}   
        self.evaluator = Evaluator(
            i3d_model_path=self.i3d_model_path,
            max_batchsize=self.max_batchsize,
            device='cuda:0'
        )
        self.val_metrics_buffer = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            x_pred, x_reconst = self.get_predictions(pl_module, batch, split="val")
            x_gt = batch['video']
            metrics = self.evaluator.evaluate_all(x_pred, x_gt, raw=True)
            # dict_keys(['mse', 'psnr', 'ssim', 'lpips', 'raw_gt_features', 'raw_pred_features', 'real_stats', 'fake_stats'])
            for key, value in metrics.items():
                if key in ["raw_gt_features", "raw_pred_features", "real_stats", "fake_stats"]:
                    # value is a list of numpy arrays
                    tensorized_value = torch.tensor(value, device=pl_module.device)
                    if key not in self.val_metrics_buffer.keys():
                        self.val_metrics_buffer[key] = tensorized_value
                    else:
                        self.val_metrics_buffer[key]= torch.cat([self.val_metrics_buffer[key], tensorized_value], dim=0)
                else:
                    if key not in self.val_metrics_buffer.keys():
                        self.val_metrics_buffer[key] = torch.tensor(value, device=pl_module.device).unsqueeze(0)
                    else:
                        self.val_metrics_buffer[key]= torch.cat([self.val_metrics_buffer[key], torch.tensor(value, device=pl_module.device).unsqueeze(0)], dim=0)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Calculate metrics for each batch during validation and average them at the end
        """
        gather_val_metrics_buffer = pl_module.all_gather(self.val_metrics_buffer)
        for key, value in gather_val_metrics_buffer.items():
            gather_val_metrics_buffer[key] = value.reshape(-1, *value.shape[2:])
        with torch.no_grad():
            if pl_module.global_rank == 0:
                for key, value in gather_val_metrics_buffer.items():
                    if key == "raw_gt_features":
                        raw_gt_features = value.cpu().numpy()
                        mu_true = np.mean(raw_gt_features, axis=0)
                        sigma_true = np.cov(raw_gt_features, rowvar=False)
                    elif key == "raw_pred_features":
                        raw_pred_features = value.cpu().numpy()
                        mu_pred = np.mean(raw_pred_features, axis=0)
                        sigma_pred = np.cov(raw_pred_features, rowvar=False)
                    elif key == "real_stats":
                        real_stats = value.cpu().numpy()
                        mu_real = np.mean(real_stats, axis=0)
                        sigma_real = np.cov(real_stats, rowvar=False)
                    elif key == "fake_stats":
                        fake_stats = value.cpu().numpy()
                        mu_fake = np.mean(fake_stats, axis=0)
                        sigma_fake = np.cov(fake_stats, rowvar=False)
                    else:
                        mean_value = torch.mean(value)
                        pl_module.logger.experiment.add_scalar(f"val/{key}", mean_value, global_step=pl_module.global_step)
                        mainlogger.info(f"Epoch end val {key}: {mean_value}")
                m = np.square(mu_pred - mu_true).sum()
                s, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
                fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))
                fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
                pl_module.logger.experiment.add_scalar(f"val/fid", fid, global_step=pl_module.global_step)
                pl_module.logger.experiment.add_scalar(f"val/fvd", fvd, global_step=pl_module.global_step)
                mainlogger.info(f"Epoch end val fvd: {fvd}")
                mainlogger.info(f"Epoch end val fid: {fid}")
            # Clear buffer and broadcast to all processes
            self.val_metrics_buffer = {}
            # Clear GPU cache after validation
            torch.cuda.empty_cache()

    def get_predictions(self, pl_module, batch, pred=None, split="train"):
        """
        run the model to get the predictions
        """
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            log_func = pl_module.log_images
            batch_logs = log_func(batch, split=split, sampled_img_num=batch['video'].shape[0], **self.log_images_kwargs)        
        for key in batch_logs:
            # dict_keys(['image_condition', 'reconst', 'condition', 'samples'])
            value = batch_logs[key]
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                ## save video grids
                video = value # b,c,t,h,w
                ## only save grayscale or rgb mode
                if video.shape[1] != 1 and video.shape[1] != 3:
                    continue
                if key == "samples":
                    x_pred = video
                elif key == "reconst":
                    x_gt = video
        if is_train:
            pl_module.train()
        return x_pred, x_gt

class CSGOLongRolloutVideoLogger(Callback):
    def __init__(self, env=None, log_every_n_train_steps=1000, buffer_size=32, i3d_model_path=None, max_batchsize=2, log_images_kwargs=None, save_dir=None, rollout_steps=100, history_steps=9, action_script="", action_repeat=1):
        super().__init__()
        self.env = env
        self.log_every_n_train_steps = log_every_n_train_steps
        self.buffer_size = buffer_size
        self.i3d_model_path = i3d_model_path
        self.max_batchsize = max_batchsize
        self.save_dir = save_dir
        self.rollout_steps = rollout_steps
        self.history_steps = history_steps
        self.action_script = action_script
        self.action_repeat = max(1, int(action_repeat))
        self.evaluate = False
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}   
        self.evaluator = Evaluator(
            i3d_model_path=self.i3d_model_path,
            max_batchsize=self.max_batchsize,
            device='cuda:0',
            env=self.env,
            save_dir=self.save_dir,
            init_eval_models=self.evaluate
        )
        self.val_metrics_buffer = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        all_exist = True
        for path in batch['path']:
            npz_file_name = os.path.splitext(os.path.basename(path))[0]
            save_folder = os.path.join(self.save_dir, npz_file_name)
            pred_path = os.path.join(save_folder, 'pred_video.mp4')
            gt_path = os.path.join(save_folder, 'gt_video.mp4')
            if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
                all_exist = False
                break
            else:
                print(f"Already exists: pred_path: {pred_path}, gt_path: {gt_path}")
        if all_exist:
            print(f"All videos already exist, skip this batch")
            return
        with torch.no_grad():
            x_pred, x_reconst = self.get_predictions(pl_module, batch, split="val")
            x_pred=x_pred.clamp(-1, 1)
            x_gt = batch['video']
            metrics = self.evaluator.evaluate_all(x_pred, x_gt, raw=True, path_dict=batch['path'], evaluate=self.evaluate)
            assert metrics is None

    def generate_random_actions(self, num_steps, device, action_unit=10):
        N_KEYS = 11
        N_CLICKS = 2
        N_MOUSE_X = 23
        N_MOUSE_Y = 15

        # If a scripted sequence is provided, use it instead of random actions.
        # Key order for first 11 dims: w,a,s,d,space,ctrl,shift,1,2,3,r
        if self.action_script:
            key_map = {
                "W": 0, "A": 1, "S": 2, "D": 3,
                " ": 4, "C": 5, "T": 6,  # C=ctrl, T=shift (to avoid clash with S key)
                "1": 7, "2": 8, "3": 9, "R": 10,
            }
            script_tokens = [ch.upper() for ch in self.action_script if ch.upper() in key_map or ch in ["_", "-", "."]]
            if not script_tokens:
                script_tokens = ["_"]

            actions_list = []
            idx = 0
            while len(actions_list) < num_steps:
                token = script_tokens[idx % len(script_tokens)]
                keys = torch.zeros(N_KEYS, device=device)
                if token in key_map:
                    keys[key_map[token]] = 1.0

                l_click = torch.zeros(1, device=device)
                r_click = torch.zeros(1, device=device)
                mouse_x = torch.zeros(N_MOUSE_X, device=device)
                mouse_x[11] = 1.0
                mouse_y = torch.zeros(N_MOUSE_Y, device=device)
                mouse_y[7] = 1.0

                action = torch.cat([keys, l_click, r_click, mouse_x, mouse_y])
                for _ in range(self.action_repeat):
                    if len(actions_list) >= num_steps:
                        break
                    actions_list.append(action)
                idx += 1

            return torch.stack(actions_list, dim=0)

        num_units = (num_steps + action_unit - 1) // action_unit
        unique_actions_list = []
        
        for _ in range(num_units):
            movement_key_idx = torch.randint(0, 4, (1,), device=device).item()
            keys = torch.zeros(N_KEYS, device=device)
            keys[movement_key_idx] = 1.0
            
            l_click = torch.zeros(1, device=device)
            r_click = torch.zeros(1, device=device)
            
            mouse_x = torch.zeros(N_MOUSE_X, device=device)
            mouse_x[11] = 1.0
            mouse_y = torch.zeros(N_MOUSE_Y, device=device)
            mouse_y[7] = 1.0
            
            action = torch.cat([keys, l_click, r_click, mouse_x, mouse_y])
            unique_actions_list.append(action)
        
        unique_actions = torch.stack(unique_actions_list, dim=0)
        
        actions_list = []
        for i in range(num_units):
            for _ in range(action_unit):
                actions_list.append(unique_actions[i])
        
        actions = torch.stack(actions_list[:num_steps], dim=0)
        
        return actions

    def get_predictions(self, pl_module, batch, pred=None, split="train"):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        torch.cuda.empty_cache()
        
        device = batch['video'].device
        B, C, T_orig, H, W = batch['video'].shape
        
        x_history = batch['video'][:, :, :self.history_steps, :, :].clone()
        action_history = batch['action'][:, :self.history_steps, :].clone()
        
        random_actions = self.generate_random_actions(self.rollout_steps, device)
        random_actions = random_actions.unsqueeze(0).expand(B, -1, -1)
        
        all_actions = torch.cat([action_history, random_actions], dim=1)
        
        log_func = pl_module.log_images
        log_kwargs = self.log_images_kwargs.copy()
        
        all_predicted_frames = []
        
        current_video_history = x_history.clone()
        
        def save_intermediate_results(step_idx, predicted_frames_list, all_actions_tensor):
            if batch.get('path') is None:
                return
            
            for b_idx in range(B):
                path = batch['path'][b_idx] if isinstance(batch['path'], list) else batch['path'][b_idx]
                npz_file_name = os.path.splitext(os.path.basename(path))[0]
                save_folder = os.path.join(self.save_dir, npz_file_name)
                os.makedirs(save_folder, exist_ok=True)
                
                if predicted_frames_list:
                    current_pred_frames = torch.cat(predicted_frames_list, dim=2)
                    current_pred_frames_b = current_pred_frames[b_idx:b_idx+1]
                else:
                    current_pred_frames_b = torch.empty(1, C, 0, H, W, device=device)
                
                history_b = x_history[b_idx:b_idx+1]
                current_video = torch.cat([history_b, current_pred_frames_b], dim=2)
                
                current_actions = all_actions_tensor[b_idx, :self.history_steps + step_idx + 1, :]
                
                def norm(t):
                    t = t.detach().cpu()
                    t = t.clamp(-1, 1)
                    return (t + 1.0) / 2.0
                
                current_video_norm = norm(current_video)
                
                video_path = os.path.join(save_folder, f'pred_video_step{step_idx+1:04d}.mp4')
                tensor_to_mp4(current_video_norm, video_path, fps=4, rescale=False)
                
                action_path = os.path.join(save_folder, f'action_step{step_idx+1:04d}.npy')
                np.save(action_path, current_actions.cpu().numpy())
                
                mainlogger.info(f"Saved intermediate results at step {step_idx+1}: {video_path}, {action_path}")
        
        for step in range(self.rollout_steps):
            placeholder_frame = current_video_history[:, :, -1:, :, :]
            step_video = torch.cat([current_video_history, placeholder_frame], dim=2)
            step_action = all_actions[:, step:step + self.history_steps + 1, :]
            
            step_batch = {
                'video': step_video,
                'action': step_action,
                'caption': batch.get('caption', [''] * B),
                'path': batch.get('path', [''] * B),
                'fps': batch.get('fps', torch.tensor([3] * B, device=device)),
                'frame_stride': batch.get('frame_stride', torch.tensor([1] * B, device=device))
            }
            
            batch_logs = log_func(
                step_batch, 
                split=split, 
                sampled_img_num=B, 
                ar=True, 
                ar_noise_schedule=1, 
                cond_frame=self.history_steps,
                **log_kwargs
            )
            
            samples = batch_logs['samples']
            next_frame = samples[:, :, -1:, :, :]
            
            current_video_history = torch.cat([current_video_history[:, :, 1:, :, :], next_frame], dim=2)
            
            all_predicted_frames.append(next_frame)
            
            if (step + 1) % 10 == 0 or step == self.rollout_steps - 1:
                save_intermediate_results(step, all_predicted_frames, all_actions)
            
            torch.cuda.empty_cache()
        
        x_pred = torch.cat(all_predicted_frames, dim=2)
        
        x_gt = x_history
        
        if is_train:
            pl_module.train()
        return x_pred, x_gt
