import hydra
import os
import torch
import time
import shutil
from omegaconf import OmegaConf
from vsr_llm import build_model
from utils import Logger,check_ckpt_path
from vsr_llm_datamodule import DataModule
from vsr_llm_modelmodule import ModelModule
from lightning import Trainer,seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@hydra.main(version_base="1.3",config_path=os.path.join(parent_dir,"conf"), config_name="configs")
def main(cfg) -> None:
    # set seed
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    TIME_STR = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # set callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint = ModelCheckpoint(
        monitor="epoch",
        mode="max",
        dirpath=os.path.join(cfg.ckpt_path, TIME_STR),
        save_last=True,
        filename="{epoch}",
        save_top_k=cfg.trainer.total_limit,
    )
    callbacks = [checkpoint, lr_monitor]

    model ,tokenizer = build_model(cfg)
    modelmodule = ModelModule(cfg, model, tokenizer)
    datamodule = DataModule(cfg, total_gpus=cfg.gpus)
    trainer = Trainer(
        accelerator="gpu",
        num_nodes=1,
        precision = cfg.trainer.mixed_precision,
        max_epochs=cfg.trainer.epochs,
        default_root_dir=cfg.ckpt_path,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.trainer.gradient_accumulation_steps,
        callbacks=callbacks,
        strategy="ddp",
        devices=cfg.gpus,
        use_distributed_sampler=False,
        log_every_n_steps=1,
        )
    trainer.validate(model=modelmodule, datamodule=datamodule,ckpt_path=cfg.pretrained_model_path)

if __name__ == "__main__":
    main()