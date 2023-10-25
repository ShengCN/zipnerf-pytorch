import os
import shutil

import accelerate
import torch
import glob
import json

def restore_checkpoint(
        checkpoint_dir,
        accelerator: accelerate.Accelerator,
        logger=None
):
    dirs = glob.glob(os.path.join(checkpoint_dir, "step-*"))
    dirs.sort()
    path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        if logger is not None:
            logger.info("Checkpoint does not exist. Starting a new training run.")
        init_step = 0
    else:
        if logger is not None:
            logger.info(f"Resuming from checkpoint {path}")
        accelerator.load_state(path)
        init_step = int(os.path.basename(path))
    return init_step


def save_checkpoint(save_dir,
                    accelerator: accelerate.Accelerator,
                    step=0,
                    total_limit=3):

    if total_limit > 0:
        folders = glob.glob(os.path.join(save_dir, "step-*"))
        folders.sort()
        for folder in folders[: len(folders) + 1 - total_limit]:
            shutil.rmtree(folder)

    accelerator.save_state(os.path.join(save_dir, f"step-{step:06d}"))


def save_best_checkpoint(save_dir, accelerator: accelerate.Accelerator, 
                         best_psnr, best_ssim, best_step):
    accelerator.save_state(os.path.join(save_dir, f"best"))
    json_file = os.path.join(save_dir, 'best.json')

    with open(json_file, 'w') as f:
        data = {
            'psnr': best_psnr,
            'ssim': best_ssim,
            'step': best_step
        }

        json.dump(data, f, indent=4)
        