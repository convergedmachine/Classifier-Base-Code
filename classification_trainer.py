# filename: classification_trainer.py

import os

import hydra
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
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


class ClassificationModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        """Initializes the classification model.

        Args:
            cfg (DictConfig): Configuration object containing hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = load_model(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(cfg)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name
        lr = self.cfg.optimizer.lr
        weight_decay = self.cfg.optimizer.weight_decay
        momentum = self.cfg.optimizer.momentum

        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported.")

        scheduler_name = self.cfg.scheduler.name
        if scheduler_name == "step":
            scheduler = MultiStepLR(
                optimizer,
                milestones=self.cfg.scheduler.lr_decay_steps,
                gamma=self.cfg.scheduler.gamma,
            )
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported.")

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


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

    backbone = load_model(cfg)
    
    # initialize backbone
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    if cfg.pretrained_backbone:
        ckpt_path = cfg.pretrained_backbone
        assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")

        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder", "backbone")] = state[k]
                logging.warn(
                    "You are using an older checkpoint. Use a new one as some issues might arrise."
                )
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]
        backbone.load_state_dict(state, strict=False)
        logging.info(f"Loaded {ckpt_path}")
    else:
        logging.info("No pretrained backbone provided, using random initialization.")    

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

    ckpt = Checkpointer(
        cfg,
        logdir=os.path.join(cfg.checkpoint.dir, "linear"),
        frequency=cfg.checkpoint.frequency,
        keep_prev=cfg.checkpoint.keep_prev,
    )
    callbacks.append(ckpt)

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

    # Initialize model
    model = ClassificationModel(cfg)

    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dir,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=cfg.checkpoint.frequency,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        devices=cfg.devices,
        accelerator=cfg.accelerator,
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.strategy == "ddp" else cfg.strategy,
        precision=cfg.precision,
        sync_batchnorm=cfg.sync_batchnorm,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        enable_checkpointing=cfg.checkpoint.enabled
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()