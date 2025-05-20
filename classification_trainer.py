import os
import inspect
import logging
import hydra

from omegaconf import DictConfig
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    OneCycleLR,
)

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import MultiStepLR

from utils.checkpointer import Checkpointer
from utils.auto_resumer import AutoResumer

from utils.dali_dataloader import ClassificationDALIDataModule
from utils.load_model import load_model
from utils.arg_parse import parse_cfg

class ClassificationModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        """
        Initializes the classification model.

        Args:
            cfg (DictConfig): Configuration object containing hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg
        self.model = load_model(cfg)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Update and log metrics
        self.train_acc.update(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.test_acc.update(preds, labels)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim_cfg = self.cfg.optimizer
        sched_cfg = self.cfg.scheduler

        # ------------------------
        # Optimizer selection
        # ------------------------
        optim_map = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'adamw': optim.AdamW,
        }
        if optim_cfg.name not in optim_map:
            raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")
        optimizer = optim_map[optim_cfg.name](
            self.parameters(),
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
            **({'momentum': optim_cfg.momentum} if optim_cfg.name == 'sgd' else {})
        )

        # ------------------------
        # Scheduler selection & parameter validation
        # ------------------------
        name = sched_cfg.name.lower()
        if name == 'step':
            if not hasattr(sched_cfg, 'step_size') or not hasattr(sched_cfg, 'gamma'):
                raise ValueError("StepLR scheduler requires 'step_size' and 'gamma'.")
            scheduler = StepLR(optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma)

        elif name == 'multistep':
            if not hasattr(sched_cfg, 'milestones') or not hasattr(sched_cfg, 'gamma'):
                raise ValueError("MultiStepLR scheduler requires 'milestones' and 'gamma'.")
            scheduler = MultiStepLR(optimizer, milestones=sched_cfg.milestones, gamma=sched_cfg.gamma)

        elif name == 'cosine':
            if not hasattr(sched_cfg, 't_max'):
                raise ValueError("CosineAnnealingLR scheduler requires 't_max'.")
            eta_min = getattr(sched_cfg, 'eta_min', 0.0)
            scheduler = CosineAnnealingLR(optimizer, T_max=sched_cfg.t_max, eta_min=eta_min)

        elif name == 'onecycle':
            oc_cfg = getattr(sched_cfg, 'onecycle', {})
            # Required OneCycleLR params: max_lr, epochs, steps_per_epoch
            max_lr = getattr(oc_cfg, 'max_lr', optim_cfg.lr)
            epochs = getattr(self.trainer, 'max_epochs', None)
            steps = getattr(self.cfg.data, 'steps_per_epoch', None)
            if epochs is None or steps is None:
                raise ValueError("OneCycleLR requires 'trainer.max_epochs' and 'data.steps_per_epoch' in config.")
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps,
                pct_start=getattr(oc_cfg, 'pct_start', 0.3),
                anneal_strategy=getattr(oc_cfg, 'anneal_strategy', 'cos'),
                div_factor=getattr(oc_cfg, 'div_factor', 25.0),
                final_div_factor=getattr(oc_cfg, 'final_div_factor', 1e4),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_cfg.name}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': sched_cfg.interval,  # 'step' or 'epoch'
                'frequency': 1,
                'monitor': 'val_loss',
            }
        }

@hydra.main(config_path="./configs/", config_name="resnet50")
def main(cfg: DictConfig):
    """Main function to set up and run the training pipeline.

    Args:
        cfg (DictConfig): Configuration object loaded from resnet50.yaml.
    """
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)

    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    # Initialize model
    model = ClassificationModel(cfg)
    
    # initialize model
    if cfg.backbone.name.startswith("resnet"):
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        if cifar:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            model.maxpool = nn.Identity()

    if cfg.pretrained_backbone:
        ckpt_path = cfg.pretrained_backbone
        assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")

        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder", "model")] = state[k]
                logging.warn(
                    "You are using an older checkpoint. Use a new one as some issues might arrise."
                )
            if "model" in k:
                state[k.replace("model.", "")] = state[k]
            del state[k]
        model.load_state_dict(state, strict=False)
        logging.info(f"Loaded {ckpt_path}")
    else:
        logging.info("No pretrained model provided, using random initialization.")    

    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, "linear"),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    wandb_logger = WandbLogger(
        name=cfg.name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        offline=cfg.wandb.offline,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
    )
    wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    # lr logging
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Initialize data module
    data_module = ClassificationDALIDataModule(
        dataset=cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        data_fraction=-1.0,  # Use all data
        dali_device="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dir,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=cfg.checkpoint.frequency,
    )
    callbacks.append(checkpoint_callback)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        devices=cfg.devices,
        accelerator=cfg.accelerator,
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.strategy == "ddp" else "auto",
        precision=cfg.precision,
        sync_batchnorm=cfg.sync_batchnorm,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=cfg.checkpoint.enabled
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()