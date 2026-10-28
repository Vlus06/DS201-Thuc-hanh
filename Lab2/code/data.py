from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _load(path, size):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))


def build_cache(data_root, split, size, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{split}_{size}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["x"], z["y"], list(z["classes"])
    ds = datasets.ImageFolder(Path(data_root) / split)
    paths = [p for p, _ in ds.samples]
    with ThreadPoolExecutor(max_workers=16) as ex:
        x = np.stack(list(ex.map(lambda p: _load(p, size), paths)))
    y = np.array(ds.targets, dtype=np.int64)
    np.savez(f, x=x, y=y, classes=np.array(ds.classes))
    return x, y, ds.classes


class CachedSet(Dataset):
    def __init__(self, x, y, idx, tf):
        self.x, self.y, self.idx, self.tf = x, y, idx, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(Image.fromarray(self.x[j])), int(self.y[j])


def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_loaders(data_root, img_size, batch_size, cache_dir, augment=True,
                  val_ratio=0.1, num_workers=8, seed=42):
    store = 256 if img_size > 128 else 128
    x, y, classes = build_cache(data_root, "train", store, cache_dir)
    xt, yt, _ = build_cache(data_root, "test", store, cache_dir)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g).tolist()
    n_val = int(len(perm) * val_ratio)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_tf, eval_tf = build_transforms(img_size)
    if not augment:
        train_tf = eval_tf
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(CachedSet(x, y, train_idx, train_tf), shuffle=True, **kw)
    val_dl = DataLoader(CachedSet(x, y, val_idx, eval_tf), shuffle=False, **kw)
    test_dl = DataLoader(CachedSet(xt, yt, list(range(len(yt))), eval_tf),
                         shuffle=False, **kw)
    return train_dl, val_dl, test_dl, classes
