import re
from typing import Callable, Sequence
import random
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import grey_dilation
from streaming import StreamingDataset
from torch import nn

COLORS = ["red", "yellow", "blue", "green"]
LOCATION_MAP = pd.read_csv("annotations/location_group_mapping.tsv", sep="\t")
UNIQUE_CATS = LOCATION_MAP["Original annotation"].unique().tolist()
UNIQUE_CATS.append("Negative")
NUM_CLASSES = len(UNIQUE_CATS)


class MinMaxNormalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min = torch.amin(x, dim=(1, 2, 3), keepdim=True)
        max = torch.amax(x, dim=(1, 2, 3), keepdim=True)

        return (x - min) / (max - min + 1e-6)


class MinMaxChannelNormalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min = torch.amin(x, dim=(2, 3), keepdim=True)
        max = torch.amax(x, dim=(2, 3), keepdim=True)

        return (x - min) / (max - min + 1e-6)


def shuffle_dict_keys(input_dict):
    keys = list(input_dict.keys())
    random.shuffle(keys)
    return {key: input_dict[key] for key in keys}


def get_stratified_idxs(df, n_idxs):
    df["basename"] = (
        df["cell_line"].astype(str) + "_" + df["plate_position"].astype(str)
    )
    index_per_category = (
        df.groupby("basename")
        .apply(lambda x: x.index.tolist(), include_groups=False)
        .to_dict()
    )

    sampled_idxs = []
    while len(sampled_idxs) < n_idxs:
        for category, indexlist in shuffle_dict_keys(index_per_category).items():
            if len(indexlist) > 0:
                random.shuffle(indexlist)
                sampled_idxs.append(indexlist.pop())
            if len(sampled_idxs) == n_idxs:
                break
    assert (
        len(sampled_idxs) == n_idxs
    ), f"n_idxs: {n_idxs}, len(sampled_idxs): {len(sampled_idxs)}"
    return sampled_idxs


class HPASubCellDatasetStream(StreamingDataset):
    def __init__(
        self,
        streams: list,
        shuffle: bool = False,
        batch_size: int = 32,
        color_channels=["red", "yellow", "blue", "green"],
        n_cells: int = -1,
        mask_prob: float = 0.0,
        normalize: str = "min_max",
        return_cell_mask: bool = False,
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
            predownload=2 * batch_size,
        )

        self.color_channels = color_channels
        self.color_idxs = [i for i, c in enumerate(COLORS) if c in color_channels]
        self.n_cells = n_cells

        self.mask_prob = mask_prob

        if normalize == "min_max":
            self.normalize = MinMaxNormalize()
        elif normalize == "min_max_channel":
            self.normalize = MinMaxChannelNormalize()
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")

        self.unique_cats = UNIQUE_CATS
        self.num_classes = NUM_CLASSES

        self.return_cell_mask = return_cell_mask

    def get_select_idxs(self, obj):
        if self.n_cells > 0:
            df = pd.DataFrame(
                {
                    "cell_line": obj["cell_line"].split(";"),
                    "plate_position": obj["plate_position"].split(";"),
                }
            )
            select_idxs = get_stratified_idxs(df, self.n_cells)
        else:
            select_idxs = np.arange(len(obj["img"]))
        return select_idxs

    def get_random_mask(self, mask_tensor):
        n_cells = mask_tensor.shape[0]
        n_non_masked = int(n_cells * (1 - self.mask_prob))
        mask_idxs = np.random.choice(n_cells, n_non_masked, replace=False)
        mask_tensor[mask_idxs] = 1.0
        return mask_tensor

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)

        select_idxs = self.get_select_idxs(obj)
        img = obj["img"][select_idxs]
        images, masks = img[:, :, :, self.color_idxs], img[:, :, :, 4]

        images_tensor = torch.from_numpy(
            images.transpose(0, 3, 1, 2).astype(np.float32)
        )
        mask_tensor = torch.from_numpy(masks[:, None, :, :].astype(np.float32))

        if self.mask_prob == 1.0:
            img_masked = images_tensor * mask_tensor
            img_normalized = self.normalize(img_masked)
            x_i = img_normalized.clone()
            x_j = img_normalized.clone()
        elif self.mask_prob == 0.0:
            img_normalized = self.normalize(images_tensor)
            x_i = img_normalized.clone()
            x_j = img_normalized.clone()
        else:
            mask_tensor1 = self.get_random_mask(mask_tensor.clone())
            img_masked1 = images_tensor * mask_tensor1
            x_i = self.normalize(img_masked1)

            mask_tensor2 = self.get_random_mask(mask_tensor.clone())
            img_masked2 = images_tensor * mask_tensor2
            x_j = self.normalize(img_masked2)

        antibody_id = obj["antibody"]
        antibody_id_tensor = torch.tensor(
            [int(re.findall(r"\d+", antibody_id)[0])] * images.shape[0]
        )
        targets = torch.tensor(obj["targets"], dtype=torch.float32)[select_idxs]

        if self.return_cell_mask:
            mask_tensor = (
                mask_tensor1
                if (self.mask_prob > 0) & (self.mask_prob < 1)
                else mask_tensor
            )
        else:
            mask_tensor = None

        return x_i, x_j, antibody_id_tensor, targets, mask_tensor


class HPASubCellDataset(StreamingDataset):
    def __init__(
        self,
        local: str,
        remote: str,
        shuffle: bool = False,
        batch_size: int = 32,
        color_channels=["red", "yellow", "blue", "green"],
        n_cells: int = -1,
        mask_prob: float = 0.0,
        normalize: str = "min_max",
        return_cell_mask: bool = False,
    ) -> None:
        super().__init__(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=batch_size,
            predownload=2 * batch_size,
            shuffle_algo="py2s",
        )

        self.color_channels = color_channels
        self.color_idxs = [i for i, c in enumerate(COLORS) if c in color_channels]
        self.n_cells = n_cells

        self.mask_prob = mask_prob

        if normalize == "min_max":
            self.normalize = MinMaxNormalize()
        elif normalize == "min_max_channel":
            self.normalize = MinMaxChannelNormalize()
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")

        self.unique_cats = UNIQUE_CATS
        self.num_classes = NUM_CLASSES

        self.return_cell_mask = return_cell_mask

        print(f"remote: {remote}", flush=True)

    def get_select_idxs(self, obj):
        if self.n_cells > 0:
            df = pd.DataFrame(
                {
                    "cell_line": obj["cell_line"].split(";"),
                    "plate_position": obj["plate_position"].split(";"),
                }
            )
            select_idxs = get_stratified_idxs(df, self.n_cells)
        else:
            select_idxs = np.arange(len(obj["img"]))
        return select_idxs

    def get_mask_tensor(self, masks):
        masks = (masks * 255).astype(np.uint8)
        masks = grey_dilation(grey_dilation(masks, size=(1, 5, 1)), size=(1, 1, 5))
        masks = (masks / 255.0).astype(np.float32)
        masks = torch.from_numpy(masks[:, None, :, :])
        return masks

    def get_random_mask(self, mask_tensor):
        n_cells = mask_tensor.shape[0]
        n_non_masked = int(n_cells * (1 - self.mask_prob))
        mask_idxs = np.random.choice(n_cells, n_non_masked, replace=False)
        mask_tensor[mask_idxs] = 1.0
        return mask_tensor

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)

        select_idxs = self.get_select_idxs(obj)
        img = obj["img"][select_idxs]
        images, masks = img[:, :, :, self.color_idxs], img[:, :, :, 4]

        images_tensor = torch.from_numpy(
            images.transpose(0, 3, 1, 2).astype(np.float32)
        )
        mask_tensor = torch.from_numpy(masks[:, None, :, :].astype(np.float32))

        if self.mask_prob == 1.0:
            img_masked = images_tensor * mask_tensor
            img_normalized = self.normalize(img_masked)
            x_i = img_normalized.clone()
            x_j = img_normalized.clone()
        elif self.mask_prob == 0.0:
            img_normalized = self.normalize(images_tensor)
            x_i = img_normalized.clone()
            x_j = img_normalized.clone()
        else:
            mask_tensor1 = self.get_random_mask(mask_tensor.clone())
            img_masked1 = images_tensor * mask_tensor1
            x_i = self.normalize(img_masked1)

            mask_tensor2 = self.get_random_mask(mask_tensor.clone())
            img_masked2 = images_tensor * mask_tensor2
            x_j = self.normalize(img_masked2)

        antibody_id = obj["antibody"]
        antibody_id_tensor = torch.tensor(
            [int(re.findall(r"\d+", antibody_id)[0])] * images.shape[0]
        )
        targets = torch.tensor(obj["targets"], dtype=torch.float32)[select_idxs]

        if self.return_cell_mask:
            mask_tensor = (
                mask_tensor1
                if (self.mask_prob > 0) & (self.mask_prob < 1)
                else mask_tensor
            )
        else:
            mask_tensor = None

        return x_i, x_j, antibody_id_tensor, targets, mask_tensor
