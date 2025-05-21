import argparse
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import timm
from utils.dali_dataloader import ImageNetDALIDataModule
from utils.resnet import resnet50, resnet50_fc, resnet50_fc_ml

def parse_args():
    parser = argparse.ArgumentParser(description="Pass config name for Hydra")
    parser.add_argument('--config-name', type=str, default='resnet50', help='Name of the Hydra config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def get_model(cfg: DictConfig, num_classes: int) -> pl.LightningModule:
    class ClassificationModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters(cfg)
            if cfg.backbone.name == "resnet50":
                self.classifier = resnet50(num_classes)
            elif cfg.backbone.name == "resnet50_fc":
                self.classifier = resnet50_fc_ml(num_classes)
            elif cfg.backbone.name == "resnet50_fc_ml":
                self.classifier = resnet50_fc_ml(num_classes)
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.classifier(x)

        def training_step(self, batch, batch_idx):
            x, y = batch[0]["data"], batch[0]["label"]
            logits = self(x)
            loss = self.criterion(logits, y.squeeze(-1))
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch[0]["data"], batch[0]["label"]
            logits = self(x)
            loss = self.criterion(logits, y.squeeze(-1))
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        def test_step(self, batch, batch_idx):
            x, y = batch[0]["data"], batch[0]["label"]
            logits = self(x)
            loss = self.criterion(logits, y.squeeze(-1))
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log("test_loss", loss, on_epoch=True)
            self.log("test_acc", acc, on_epoch=True)

        def configure_optimizers(self):
            opt_cfg = cfg.optimizer
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=opt_cfg.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay
            )
            sched_cfg = cfg.scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': sched_cfg.interval
                }
            }

    return ClassificationModel()

def dynamic_main(config_name):
    @hydra.main(config_path="configs", config_name=config_name, version_base="1.1")
    def main(cfg: DictConfig):
        pl.seed_everything(getattr(cfg, 'seed', 42))
        datamodule = ImageNetDALIDataModule(
            cfg.data.train_path,
            cfg.data.val_path,
            cfg.data.get('test_path', None),
            cfg.optimizer.batch_size,
            num_threads=cfg.data.num_workers,
            data_fraction=cfg.data.data_fraction
        )
        datamodule.setup('fit')
        num_classes = datamodule.num_classes
        model = get_model(cfg, num_classes)
        wandb_run_id = None
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.name,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,        
        )
        callbacks = []
        if cfg.checkpoint.enabled:
            ckpt_cb = ModelCheckpoint(
                dirpath=cfg.checkpoint.dir,
                filename="{epoch}",
                every_n_epochs=cfg.checkpoint.frequency,
                save_top_k=-1,
                save_last=True
            )
            callbacks.append(ckpt_cb)
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            devices=cfg.devices,
            accelerator=cfg.accelerator,
            strategy=cfg.strategy,
            precision=cfg.precision,
            sync_batchnorm=cfg.sync_batchnorm,
            logger=wandb_logger,
            callbacks=callbacks
        )
        trainer.fit(model, datamodule)
        if hasattr(datamodule, 'test_dataloader'):
            trainer.test(model, datamodule)
    
    return main  # Return the decorated main function

if __name__ == "__main__":
    args = parse_args()
    main_fn = dynamic_main(args.config_name)
    main_fn()