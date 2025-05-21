import os
from pathlib import Path
from typing import List, Union

import torch
import pytorch_lightning as pl

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali import fn, types


def get_dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return world_size, rank


class ImageNetDALIPipeline(Pipeline):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_shards: int,
        shard_id: int,
        device_id: int,
        seed: int = 42,
        num_threads: int = 4,
        crop_size: int = 224,
        is_training: bool = True,
    ):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )
        self.data_dir = data_dir
        self.is_training = is_training
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.crop_size = crop_size
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    def define_graph(self):
        # 1) Read & shard
        jpegs, labels = fn.readers.file(
            name="Reader",
            file_root=self.data_dir,
            random_shuffle=self.is_training,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
        )
        labels = fn.cast(labels, dtype=types.INT64)

        # 2) Decode
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        if self.is_training:
            # 3a) RandomResizedCrop + flip
            images = fn.random_resized_crop(
                images,
                size=(self.crop_size, self.crop_size),
                random_area=(0.08, 1.0),
                random_aspect_ratio=(3.0/4.0, 4.0/3.0),
            )
            mirror = fn.random.coin_flip(probability=0.5)
            out = fn.crop_mirror_normalize(
                images,
                device="gpu",
                output_layout=types.NCHW,
                mean=self.mean,
                std=self.std,
                mirror=mirror,
            )
        else:
            # 3b) Resize shorter side + center crop
            images = fn.resize(
                images,
                device="gpu",
                resize_shorter=256,
                interp_type=types.INTERP_TRIANGULAR,
            )
            out = fn.crop_mirror_normalize(
                images,
                device="gpu",
                output_layout=types.NCHW,
                crop=(self.crop_size, self.crop_size),
                mean=self.mean,
                std=self.std,
            )

        return out, labels


def _collect_labels(*paths: Union[str, Path]) -> List[str]:
    labels = set()
    for p in paths:
        for entry in os.scandir(p):
            if entry.is_dir():
                labels.add(entry.name)
    return sorted(labels)


def get_dali_dataloader(
    data_dir: str,
    batch_size: int,
    training: bool = True,
    num_threads: int = 4,
    data_fraction: float = 1.0,
):
    world_size, rank = get_dist_info()

    # count total images
    total = 0
    for root, _, files in os.walk(data_dir):
        total += sum(1 for f in files if f.lower().endswith((".jpg", "jpeg", ".png")))

    # optionally limit to a fraction
    if training and 0 < data_fraction < 1.0:
        sample_count = max(int(total * data_fraction), batch_size)
    else:
        sample_count = None

    # build & run pipeline
    pipe = ImageNetDALIPipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        num_shards=world_size,
        shard_id=rank,
        device_id=rank,
        num_threads=num_threads,
        is_training=training,
    )
    pipe.build()

    # branch to avoid passing size + reader_name together
    if sample_count is not None:
        return DALIClassificationIterator(
            pipelines=[pipe],
            size=sample_count,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
    else:
        return DALIClassificationIterator(
            pipelines=[pipe],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )


class ImageNetDALIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        batch_size: int,
        num_threads: int = 4,
        data_fraction: float = 1.0,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.data_fraction = data_fraction
        self.num_classes = len(_collect_labels(train_dir))

    def train_dataloader(self):
        return get_dali_dataloader(
            data_dir=self.train_dir,
            batch_size=self.batch_size,
            training=True,
            num_threads=self.num_threads,
            data_fraction=self.data_fraction,
        )

    def val_dataloader(self):
        return get_dali_dataloader(
            data_dir=self.val_dir,
            batch_size=self.batch_size,
            training=False,
            num_threads=self.num_threads,
        )

    def test_dataloader(self):
        return get_dali_dataloader(
            data_dir=self.test_dir,
            batch_size=self.batch_size,
            training=False,
            num_threads=self.num_threads,
        )
