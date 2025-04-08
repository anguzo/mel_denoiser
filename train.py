import argparse
import os

import torch

torch.set_float32_matmul_precision("high")

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.callbacks import LoggingCallback
from src.dataset import MelDenoiserDataModule
from src.model import MelDenoiser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MelDenoiser")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--last_ckpt",
        type=str,
        default=None,
        help="Path to the last checkpoint (optional)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate of the audio files",
    )
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "default")

    logger = TensorBoardLogger("lightning_logs", name="", version=f"{job_id}")

    early_stopping = EarlyStopping(
        monitor="valid/loss",
        patience=10,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/loss",
        dirpath=f"lightning_logs/{job_id}/ckpts",
        filename="mel_denoiser-{epoch}-{valid/loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    logging_callback = LoggingCallback(sample_rate=args.sample_rate)

    model = MelDenoiser()

    data_module = MelDenoiserDataModule(
        data_dir=args.data_dir,
        batch_size=32,
        num_workers=8,
        sample_rate=args.sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    trainer = L.Trainer(
        max_steps=1_500_000,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback, logging_callback],
    )

    trainer.fit(model, data_module, ckpt_path=args.last_ckpt)
    trainer.test(
        model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
    )
