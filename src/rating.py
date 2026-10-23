import os
import math
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
)


# ── ImageNet normalisation (fixed standard values) ────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
# 4-band variant for the 16-bit R,G,B,NIR SR tiles: ImageNet RGB + a neutral NIR.
IMAGENET_MEAN_4 = (0.485, 0.456, 0.406, 0.450)
IMAGENET_STD_4  = (0.229, 0.224, 0.225, 0.225)
UINT16_MAX = 65535.0


def read_multiband(path: str, in_channels: int) -> np.ndarray:
    """Load a 16-bit multiband TIFF as float32 (H, W, in_channels) in [0, 1].

    Mirrors neosr's read (uint16 / 65535) and tolerates planar (C,H,W) layout."""
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    arr = arr[..., :in_channels].astype(np.float32)
    denom = 255.0 if arr.dtype == np.uint8 else UINT16_MAX
    return arr / denom

# ── Channel counts per backbone ───────────────────────────────────────
_RESNET_CHANNELS   = {"resnet18": 512, "resnet50": 2048, "resnet152": 2048}
_DENSENET_CHANNELS = {"densenet121": 1024, "densenet201": 1920}
# timm backbones: final spatial feature-map channels before global pooling
_TIMM_CHANNELS = {
    "darknet53":           1024,   # DarkNet-53 (YOLOv3 backbone)
    "inception_resnet_v2": 1536,   # Inception-ResNet-v2 (Szegedy et al., 2017)
    "xception":            2048,   # Xception (Chollet, 2017)
}
# ViT: timm model names (no spatial map → HVCA not applicable)
# Pretrained positional embeddings are interpolated to the actual img_size at load time.
# Non-multiples of patch_size (16) are handled by floor division; edge pixels are dropped.
_VIT_TIMM_NAMES = {
    "vit_b_16": "vit_base_patch16_224",    # ViT-Base  / patch 16
    "vit_l_16": "vit_large_patch16_224",   # ViT-Large / patch 16
}


# =========================
# Experiment helpers
# =========================

class Logger:
    """Writes messages to stdout and to a log file simultaneously."""

    def __init__(self, log_path):
        self.log_path = log_path
        open(log_path, "w", encoding="utf-8").close()

    def log(self, msg: str = "") -> None:
        try:
            print(msg)
        except UnicodeEncodeError:
            # console may be cp1252 (Windows); the log file is utf-8 and keeps the original
            print(msg.encode("ascii", "replace").decode("ascii"))
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def create_experiment_folder(experiments_dir: str) -> Path:
    """Create a timestamped experiment directory under *experiments_dir*."""
    tz = ZoneInfo("Europe/Madrid")
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    folder = Path(experiments_dir) / f"experiment_{ts}_CET"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# =========================
# Dataset
# =========================

# Decoded tiles, kept in memory for the life of the process. Decoding the PNG was the largest
# part of loading (0.66 of 1.4 ms per tile, measured 2026-10-02), and the same tiles are read
# again at every epoch and every check. A tile is now read and decoded once; later epochs,
# checks and runs in the same process get the identical pixels from RAM. The tiles are small
# (202x54 px RGB, ~33 KB decoded), so all 5,359 tiles of one input take ~175 MB.
_TILE_CACHE: dict = {}


def load_rgb_cached(fpath: str) -> Image.Image:
    """The tile at *fpath* as an RGB PIL image, decoded from disk only the first time.
    The cached image is shared: transforms must return new images (torchvision's do)."""
    im = _TILE_CACHE.get(fpath)
    if im is None:
        with Image.open(fpath) as f:
            im = f.convert("RGB")          # convert() returns a new, fully loaded image
        _TILE_CACHE[fpath] = im
    return im


class RatingsDataset(Dataset):
    """CSV-backed dataset of road-segment images with float quality ratings.

    Each CSV row must have at least 'filepath' (relative to *data_root* or
    absolute) and 'rating' (float, 1–5).

    Optional per-sample augmentation: if *augment_transform* is given and the
    rounded rating belongs to *augmented_classes*, the augmentation transform
    is used instead of the base *transform*.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        transform=None,
        class_weights=None,
        augment_transform=None,
        augmented_classes=None,
        in_channels=3,
    ):
        self.csv_path         = csv_path
        self.data_root        = data_root
        self.transform        = transform
        self.class_weights    = class_weights
        self.augment_transform = augment_transform
        self.augmented_classes = augmented_classes or []
        self.in_channels      = in_channels   # 3 -> PIL RGB (8-bit); 4 -> 16-bit R,G,B,NIR TIFF
        self.df = pd.read_csv(self.csv_path)
        required_cols = {"filepath", "rating"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        # full paths and ratings as plain lists: a pandas row lookup per tile is slow. The CSVs
        # store Windows paths (ROI\ROAD\tile.png); on Linux the separators are converted.
        fps = self.df["filepath"]
        if os.sep == "/":
            fps = fps.str.replace("\\", "/", regex=False)
        self._paths = [f if os.path.isabs(f) else os.path.join(self.data_root, f) for f in fps]
        self._ratings = [float(r) for r in self.df["rating"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fpath  = self._paths[idx]
        rating = self._ratings[idx]

        use_aug = (self.augment_transform is not None
                   and int(round(rating)) in self.augmented_classes)

        if self.in_channels != 3:
            # 16-bit multiband TIFF -> [C,H,W] float tensor; transforms operate on tensors
            arr = read_multiband(fpath, self.in_channels)        # (H, W, C) in [0, 1]
            img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            tfm = self.augment_transform if use_aug else self.transform
            if tfm is not None:
                img = tfm(img)
        else:
            im = load_rgb_cached(fpath)
            if use_aug:
                img = self.augment_transform(im)
            elif self.transform is not None:
                img = self.transform(im)
            else:
                img = im.copy()               # never hand out the shared cached image

        rating_tensor = torch.tensor(rating, dtype=torch.float32)

        if self.class_weights is not None:
            weight = self.class_weights.get(int(round(rating)), 1.0)
            weight_tensor = torch.tensor(weight, dtype=torch.float32)
        else:
            weight_tensor = torch.tensor(1.0, dtype=torch.float32)

        return img, rating_tensor, weight_tensor


def build_transforms(img_height: int, img_width: int, in_channels: int = 3) -> transforms.Compose:
    if in_channels == 3:
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    # 16-bit multiband path: input is already a [C,H,W] float tensor in [0,1]
    return transforms.Compose([
        transforms.Resize((img_height, img_width), antialias=True),
        transforms.Normalize(mean=IMAGENET_MEAN_4[:in_channels], std=IMAGENET_STD_4[:in_channels]),
    ])


def build_augmented_transforms(img_height: int, img_width: int, in_channels: int = 3) -> transforms.Compose:
    """Flips for minority classes (colour jitter only in the 3-band PIL path)."""
    if in_channels == 3:
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    # 16-bit multiband path (tensor in): flips are channel-agnostic; ColorJitter is PIL-only so omitted
    return transforms.Compose([
        transforms.Resize((img_height, img_width), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=IMAGENET_MEAN_4[:in_channels], std=IMAGENET_STD_4[:in_channels]),
    ])


def calculate_class_weights(train_csv: str, mode: str = "squared") -> dict:
    """Per-class loss weights from the training label counts.

    mode "inverse": w = N / (K * n_c), the standard "balanced" weights (scikit-learn,
        after King & Zeng 2001). Every class then carries the same total weight, so the
        weighted training loss equals the loss averaged within each class and then across
        classes: the same grade-averaged quantity used for early stopping and evaluation.
    mode "squared": w squared (the original behaviour, kept so archived runs reproduce).
    """
    assert mode in ("inverse", "squared"), mode
    df = pd.read_csv(train_csv)
    ratings      = df["rating"].round().astype(int)
    class_counts = ratings.value_counts().to_dict()
    total        = len(df)
    n_classes    = len(class_counts)
    weights = {}
    for r in range(1, 6):
        count     = class_counts.get(r, 1)
        w         = total / (n_classes * count)
        weights[r] = w if mode == "inverse" else w * w
    return weights


def compute_oversampled_classes(csv_path: str) -> List[int]:
    """Return class IDs whose sample count is below the per-class average.

    These classes will be drawn more than once per epoch on average when
    WeightedRandomSampler is used, so they need augmentation to avoid
    pixel-identical repeated samples.
    """
    df = pd.read_csv(csv_path)
    ratings      = df["rating"].round().astype(int)
    class_counts = ratings.value_counts().to_dict()
    total        = len(df)
    n_classes    = len(class_counts)
    avg          = total / n_classes
    return sorted(cls for cls, cnt in class_counts.items() if cnt < avg)


def get_class_counts(csv_path: str, num_classes: int = 5) -> dict:
    """Return {class_id: count} from a ratings CSV (true counts, not sampled)."""
    df = pd.read_csv(csv_path)
    counts = df["rating"].round().astype(int).value_counts().to_dict()
    return {c: counts.get(c, 0) for c in range(1, num_classes + 1)}


def build_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that gives each class equal draw probability.

    Per-sample weight = 1 / class_count.  num_samples = len(dataset) so
    each epoch sees the same number of iterations as without oversampling.
    """
    ratings      = dataset.df["rating"].round().astype(int).tolist()
    class_counts = {}
    for r in ratings:
        class_counts[r] = class_counts.get(r, 0) + 1
    weights = [1.0 / class_counts[r] for r in ratings]
    return WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
    )


# =========================
# HV Channel Attention
# =========================

class HVChannelAttention(nn.Module):
    """Horizontal-Vertical Channel Attention.

    Compresses the feature map along rows (L∞) and columns (L1), fuses
    the two descriptors, and produces per-channel scaling weights via an MLP.

    Input:  x  [B, C, H, W]
    Output: x * weights  (channel-wise scaling, same shape)
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 4):
        super().__init__()
        mid = max(1, num_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        col_max = x.abs().max(dim=2).values   # [B, C, W]
        v_vec   = col_max.mean(dim=2)          # [B, C]
        row_l1  = x.abs().mean(dim=3)          # [B, C, H]
        h_vec   = row_l1.mean(dim=2)           # [B, C]
        fused   = torch.cat([v_vec, h_vec], dim=1)
        return x * self.mlp(fused).unsqueeze(-1).unsqueeze(-1)


class ResNetWithHVCA(nn.Module):
    """ResNet backbone with HV Channel Attention inserted before the classifier."""

    def __init__(self, backbone: nn.Module, backbone_name: str, reduction_ratio: int = 4):
        super().__init__()
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.hv_attention = HVChannelAttention(_RESNET_CHANNELS[backbone_name], reduction_ratio)
        self.avgpool = backbone.avgpool
        self.fc      = backbone.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x);   x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x);  x = self.layer2(x)
        x = self.layer3(x);  x = self.layer4(x)
        x = self.hv_attention(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


class DenseNetWithHVCA(nn.Module):
    """DenseNet backbone with HV Channel Attention inserted before the classifier."""

    def __init__(self, backbone: nn.Module, backbone_name: str, reduction_ratio: int = 4):
        super().__init__()
        self.features     = backbone.features
        self.hv_attention = HVChannelAttention(_DENSENET_CHANNELS[backbone_name], reduction_ratio)
        self.classifier   = backbone.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.features(x))
        x = self.hv_attention(x)
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1))


class TimmBackboneWithHVCA(nn.Module):
    """Generic wrapper for timm backbones with optional HV Channel Attention.

    Uses timm's global_pool='' to preserve the spatial feature map [B, C, H, W],
    optionally applies HVCA, then global average pools and passes through a
    linear regression head.

    BACKBONE_BLOCKS semantics differ from ResNet/DenseNet: the backbone is a
    single child ('features'), plus optionally 'hv_attention'.
        0 → head only
        1 → full backbone (+ hv_attention if present)

    Requires: pip install timm
    """

    def __init__(
        self,
        backbone_name: str,
        num_features: int,
        use_attention: bool = False,
        reduction_ratio: int = 4,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError(
                f"timm is required for '{backbone_name}'. Install with: pip install timm"
            )
        # global_pool='' keeps the spatial feature map so HVCA can be applied
        self.features = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool=""
        )
        if use_attention:
            self.hv_attention = HVChannelAttention(num_features, reduction_ratio)
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if hasattr(self, "hv_attention"):
            x = self.hv_attention(x)
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class TimmViTRegressor(nn.Module):
    """ViT backbone via timm for regression on non-standard image sizes.

    timm handles non-square, non-multiples-of-patch-size inputs by:
      - floor(H / patch_size) × floor(W / patch_size) patches (edge pixels dropped)
      - pretrained positional embeddings interpolated to the actual patch grid

    HVCA is not applicable (ViT operates on patch tokens, not spatial feature maps).

    BACKBONE_BLOCKS semantics: 0 = head only, ≥1 = full backbone fine-tuning.
    Individual transformer block control is not supported via this wrapper.

    Requires: pip install timm
    """

    def __init__(self, timm_name: str, img_size: tuple, use_attention: bool = False):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for ViT. Install with: pip install timm")
        if use_attention:
            print("WARNING: HVCA is not applicable to ViT (patch tokens, no spatial map). Ignoring.")
        # num_classes=0 removes the pretrained head; img_size triggers pos-embed interpolation
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0,
                                          img_size=img_size)
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


# =========================
# Model building
# =========================

def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def get_head_module(model: nn.Module) -> Tuple[Optional[str], Optional[nn.Module]]:
    if hasattr(model, "fc"):
        return "fc", model.fc
    if hasattr(model, "classifier"):
        return "classifier", model.classifier
    children = list(model.named_children())
    if children:
        name, module = children[-1]
        return name, module
    return None, None


def architecture_stages(model: nn.Module, backbone: str):
    """(container, all child names, stage names) for architectures with real stages.

    A "stage" is a resolution level of the backbone: denseblock1..4 for DenseNet,
    layer1..4 for ResNet. Returns (None, None, None) when the notion does not apply, in
    which case only "head" and "all" are meaningful."""
    if backbone.startswith("densenet"):
        names = [n for n, _ in model.features.named_children()]
        return model.features, names, [n for n in names if n.startswith("denseblock")]
    if backbone.startswith("resnet"):
        names = [n for n, _ in model.named_children()]
        return model, names, [n for n in names if n.startswith("layer")]
    return None, None, None


def unfreeze_by_spec(model: nn.Module, backbone: str, spec: str) -> List[str]:
    """Unfreeze part of the network by an explicit, architecture-independent spec.

        "head"   only the regression head
        "lastN"  the last N stages of the backbone, plus everything after them
        "all"    the whole backbone

    This replaces the integer BACKBONE_BLOCKS, whose meaning silently depended on how a
    backbone happened to nest its modules: torchvision ResNet exposes layer1..layer4 as
    top-level children, but DenseNet, Xception and DarkNet wrap the entire backbone in a
    single `features` Sequential, so any value >= 1 unfroze everything and values 1, 2 and
    4 were indistinguishable. Stages here are the architecture's real ones, so "last1"
    means the same thing for every backbone that has stages at all.

    Note the stage boundary is taken *inclusive of everything after it*: "last1" on a
    DenseNet is denseblock4 + norm5, not denseblock4 alone, because the trailing norm
    belongs to that resolution level."""
    head_name, head_module = get_head_module(model)
    freeze_all(model)
    if head_module is not None:
        for p in head_module.parameters():
            p.requires_grad = True

    if spec == "head":
        return [head_name] if head_name else []

    if spec == "all":
        for name, module in model.named_children():
            if name != head_name:
                for p in module.parameters():
                    p.requires_grad = True
        return ["<backbone>"] + ([head_name] if head_name else [])

    if not spec.startswith("last"):
        raise ValueError(f"unknown unfreeze spec {spec!r} (want 'head', 'all' or 'lastN')")
    n = int(spec[4:])
    container, names, stages = architecture_stages(model, backbone)
    if container is None:
        raise ValueError(
            f"unfreeze spec {spec!r} is not defined for backbone {backbone!r}: it exposes "
            f"no identifiable stages, so only 'head' and 'all' are meaningful")
    if n > len(stages):
        raise ValueError(f"{backbone} has {len(stages)} stages, cannot unfreeze last {n}")
    start = names.index(stages[-n])
    for name in names[start:]:
        for p in getattr(container, name).parameters():
            p.requires_grad = True
    return names[start:] + ([head_name] if head_name else [])


def unfreeze_last_n_blocks(
    model: nn.Module,
    n_backbone_blocks: int = 0,
    always_train_head: bool = True,
) -> List[str]:
    """LEGACY: unfreeze the last n TOP-LEVEL children. Kept so archived experiment
    configurations keep reproducing exactly. New work should use unfreeze_by_spec."""
    freeze_all(model)
    unfrozen = []
    head_name, head_module = get_head_module(model)

    children          = list(model.named_children())
    backbone_children = [(n, m) for (n, m) in children if n != head_name] if head_name else children

    if n_backbone_blocks > 0 and backbone_children:
        n = min(n_backbone_blocks, len(backbone_children))
        if n < n_backbone_blocks:
            print(f"WARNING: asked to unfreeze {n_backbone_blocks} backbone blocks but this "
                  f"model exposes only {len(backbone_children)} top-level backbone "
                  f"module(s) {[c[0] for c in backbone_children]}; unfreezing all of them. "
                  f"Values above {len(backbone_children)} are indistinguishable here.")
        for name, module in backbone_children[-n:]:
            for p in module.parameters():
                p.requires_grad = True
            unfrozen.append(name)

    if always_train_head and head_module is not None:
        for p in head_module.parameters():
            p.requires_grad = True
        if head_name:
            unfrozen.append(head_name)

    return unfrozen


def inflate_first_conv(model: nn.Module, in_channels: int) -> str:
    """Replace the model's first Conv2d to accept `in_channels` instead of 3.

    Pretrained RGB filters are copied; any extra channel (NIR) is initialised to the
    mean of the RGB filters. Returns the top-level module name to keep trainable."""
    first_name, first = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_name, first = name, module
            break
    if first is None or first.in_channels == in_channels:
        return first_name or ""

    new = nn.Conv2d(in_channels, first.out_channels, first.kernel_size,
                    stride=first.stride, padding=first.padding,
                    dilation=first.dilation, groups=first.groups,
                    bias=first.bias is not None)
    with torch.no_grad():
        w = first.weight                      # [out, c_in, kh, kw]
        c = min(in_channels, w.shape[1])
        new.weight[:, :c] = w[:, :c]
        if in_channels > w.shape[1]:
            new.weight[:, w.shape[1]:] = w.mean(dim=1, keepdim=True).repeat(
                1, in_channels - w.shape[1], 1, 1)
        if first.bias is not None:
            new.bias.copy_(first.bias)

    parent, parts = model, first_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)
    for p in new.parameters():
        p.requires_grad = True
    return parts[0]


def build_model(
    backbone: str,
    backbone_blocks,          # int (legacy top-level children) | str ("head"/"lastN"/"all")
    use_channel_attention: bool,
    attention_reduction_ratio: int = 4,
    img_size: tuple = (224, 224),
    in_channels: int = 3,
    head_dropout: float = 0.0,
) -> Tuple[nn.Module, List[str]]:
    """Build backbone, replace head, apply freezing, optionally wrap with HV-CA.

    head_dropout > 0 puts a Dropout on the pooled feature vector right before the final
    linear layer (as torchvision's EfficientNet head does). The head keeps its attribute
    name, so freezing and "head only" training treat it exactly as before.

    Supported backbones
    -------------------
    torchvision : resnet18, resnet50, resnet152, densenet121, densenet201
    timm        : darknet53, inception_resnet_v2, xception  (pip install timm)
                  vit_b_16, vit_l_16                        (pip install timm)

    HVCA notes
    ----------
    - ResNets / DenseNets : wrapped post-hoc with ResNetWithHVCA / DenseNetWithHVCA
    - timm CNN backbones  : HVCA integrated inside TimmBackboneWithHVCA
    - ViTs                : HVCA not applicable (patch tokens, no spatial map)

    img_size : (H, W) — only used for ViT to interpolate positional embeddings.

    Returns (model, unfrozen_module_names).
    """
    model = None

    # ── torchvision ResNets ──────────────────────────────────────────────
    if backbone == "resnet18":
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = "IMAGENET1K_V1"
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)

    if backbone == "resnet50":
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)

    if backbone == "resnet152":
        try:
            weights = models.ResNet152_Weights.IMAGENET1K_V2
            model = models.resnet152(weights=weights)
        except AttributeError:
            model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)

    # ── torchvision DenseNets ────────────────────────────────────────────
    if backbone == "densenet121":
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = "IMAGENET1K_V1"
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    if backbone == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    # ── timm CNN backbones (HVCA integrated directly in the wrapper) ─────
    if backbone in _TIMM_CHANNELS:
        model = TimmBackboneWithHVCA(
            backbone, _TIMM_CHANNELS[backbone],
            use_attention=use_channel_attention,
            reduction_ratio=attention_reduction_ratio,
        )

    # ── timm ViTs (pos embeddings interpolated to img_size at load time) ─
    if backbone in _VIT_TIMM_NAMES:
        model = TimmViTRegressor(
            _VIT_TIMM_NAMES[backbone], img_size=img_size,
            use_attention=use_channel_attention,
        )

    if model is None:
        valid = (list(_RESNET_CHANNELS) + list(_DENSENET_CHANNELS)
                 + list(_TIMM_CHANNELS) + list(_VIT_TIMM_NAMES))
        raise ValueError(f"Unsupported backbone: {backbone!r}. Valid: {valid}")

    # Optional dropout on the input of the regression head (the pooled features).
    if head_dropout > 0:
        head_name, head_module = get_head_module(model)
        assert isinstance(head_module, nn.Linear), f"head is {type(head_module).__name__}"
        setattr(model, head_name, nn.Sequential(nn.Dropout(head_dropout), head_module))

    # A string selects the stage-aware path; an int keeps the legacy top-level-children
    # behaviour so archived experiment configurations still reproduce exactly.
    if isinstance(backbone_blocks, str):
        unfrozen = unfreeze_by_spec(model, backbone, backbone_blocks)
    else:
        unfrozen = unfreeze_last_n_blocks(
            model, n_backbone_blocks=backbone_blocks, always_train_head=True
        )

    # ── Adapt the stem to 4-band (R,G,B,NIR) input; new conv stays trainable ──
    if in_channels != 3:
        unfrozen = unfrozen + [inflate_first_conv(model, in_channels)]

    # ── Post-hoc HVCA wrapping (ResNets and DenseNets only) ─────────────
    # timm CNN models already have HVCA baked in; ViTs don't support it.
    if use_channel_attention:
        if backbone in _RESNET_CHANNELS:
            model = ResNetWithHVCA(model, backbone, attention_reduction_ratio)
            unfrozen = unfrozen + ["hv_attention"]
        elif backbone in _DENSENET_CHANNELS:
            model = DenseNetWithHVCA(model, backbone, attention_reduction_ratio)
            unfrozen = unfrozen + ["hv_attention"]

    return model, unfrozen


# =========================
# Loss functions
# =========================

class BONNThresholds(nn.Module):
    """M-1 learnable decision thresholds for BONN, guaranteed strictly ordered.

    Parameterised as a base threshold u_0 plus M-2 positive increments via
    softplus, so u_0 < u_1 < ... < u_{M-2} is enforced by construction.
    """

    def __init__(self, num_classes: int = 5, init_spacing: float = 0.5):
        super().__init__()
        M = num_classes
        # Centre initial thresholds symmetrically around 0
        u0_init = -((M - 2) / 2.0) * init_spacing
        self.u0 = nn.Parameter(torch.tensor(u0_init))
        # Initialise increments via softplus^{-1}(init_spacing)
        inv_sp = math.log(math.exp(init_spacing) - 1.0)
        self.log_deltas = nn.Parameter(torch.full((M - 2,), inv_sp))

    def forward(self) -> torch.Tensor:
        """Return ordered thresholds [M-1]."""
        increments = F.softplus(self.log_deltas)           # [M-2], all positive
        return self.u0 + torch.cat([
            torch.zeros(1, device=self.u0.device),
            torch.cumsum(increments, dim=0),
        ])

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Threshold-based class prediction: d if z ∈ (u_{d-1}, u_d]. Returns [B], 1-indexed."""
        u = self.forward()
        pred = torch.ones(z.shape[0], dtype=torch.long, device=z.device)
        for threshold in u:
            pred += (z > threshold).long()
        return pred.clamp(1, len(u) + 1)


class BONNLoss(nn.Module):
    """Bayesian Ordinal Neural Network loss (Lázaro & Figueiras-Vidal, PR 2023).

    Minimises an estimate of the Bayes classification cost using Parzen windows
    over the scalar network output. Handles ordinal structure and class imbalance
    jointly in the loss function.

    Two modes:
      "mae"  — π_t = N_t/N: each sample equally weighted; optimises MAE
      "amae" — π_t = 1/M:   each class equally weighted; optimises AMAE;
               recommended for imbalanced datasets (equivalent to using
               costs c_{d,t} = |d-t| / π_t, see Eq. 42 in the paper)

    Cost matrix: c_{d,t} = |d - t|  (ordinal absolute error).
    Parzen window: Gaussian with std dev σ (paper default: σ = √0.1507 ≈ 0.388).

    Args:
        num_classes : M — number of ordinal classes
        mode        : "mae" or "amae"
        sigma       : Gaussian window std dev

    Inputs to forward():
        z            [B]     scalar network outputs (raw, pre-threshold)
        targets      [B]     float ratings (rounded internally to integer class)
        thresholds   [M-1]   ordered thresholds from BONNThresholds.forward()
        class_counts dict    {class_id: count} from the training set

    Normalisation (added after an audit against the authors' reference code,
    MLPBayesian-Python, class MLPOrdinal):
        "batch"  : the reference estimate. For each class present in the batch, the cost is
                   averaged over THAT class's samples in the batch and multiplied by its prior
                   (1/M in "amae" mode, N_t/N in "mae" mode). Needs every class in every batch
                   (see RepresentativeBatchSampler).
        "legacy" : the original implementation, which divides by the class count of the whole
                   TRAINING set; the loss is then ~batch/N times too small and very noisy.
                   Kept only so archived runs reproduce.
    """

    def __init__(self, num_classes: int = 5, mode: str = "amae", sigma: float = 0.3883,
                 normalisation: str = "legacy"):
        super().__init__()
        assert normalisation in ("legacy", "batch"), normalisation
        self.M     = num_classes
        self.mode  = mode
        self.sigma = sigma
        self.normalisation = normalisation

    def _normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian CDF K(x) = Φ(x / σ)."""
        return 0.5 * (1.0 + torch.erf(x / (self.sigma * math.sqrt(2.0))))

    def forward(
        self,
        z:            torch.Tensor,
        targets:      torch.Tensor,
        thresholds:   torch.Tensor,
        class_counts: dict,
    ) -> torch.Tensor:
        M = self.M
        N = z.shape[0]
        t_int = targets.round().long().clamp(1, M)   # [B]

        # cost_diff[d, k] = c_{d,t_k} - c_{d+1,t_k} = |d - t_k| - |d+1 - t_k|
        # = -1 if t_k <= d, +1 if t_k > d
        d_vals = torch.arange(1, M, dtype=torch.float32, device=z.device)  # [M-1]
        t_f    = t_int.float()                                              # [B]
        cost_diff = (                                                        # [M-1, B]
            (d_vals.unsqueeze(1) - t_f.unsqueeze(0)).abs()
            - (d_vals.unsqueeze(1) + 1 - t_f.unsqueeze(0)).abs()
        )

        # K(u_d - z_k) for all threshold-sample pairs: [M-1, B]
        K_vals = self._normal_cdf(thresholds.unsqueeze(1) - z.unsqueeze(0))
        per_sample = (cost_diff * K_vals).sum(0)                           # [B]

        if self.normalisation == "batch":
            # reference: sum over classes of prior_t * mean over the batch's samples of class t.
            # The constant |M - t| (cost of deciding the top grade) is added back, so the value
            # is the FULL expected cost in grades (a smoothed MAE / AMAE, always >= 0) instead
            # of that cost minus a constant; gradients and every comparison are unchanged.
            total_train = sum(class_counts.values())
            loss = z.new_zeros(())
            for c in torch.unique(t_int):
                prior = (1.0 / M if self.mode == "amae"
                         else class_counts.get(int(c), 0) / max(total_train, 1))
                loss = loss + prior * (per_sample[t_int == c].mean() + (M - int(c)))
            return loss

        # Per-sample normalisation weight (legacy)
        if self.mode == "mae":
            weights = torch.full((N,), 1.0 / max(N, 1), device=z.device)  # [B]
        else:  # "amae": weight by 1/N_t — all classes contribute equally
            n_t = torch.tensor(
                [max(class_counts.get(int(c), 1), 1) for c in t_int.tolist()],
                dtype=torch.float32, device=z.device,
            )  # [B]
            weights = 1.0 / n_t

        # BC_hat = Σ_{d,k} cost_diff[d,k] * weights[k] * K[d,k]
        return (per_sample * weights).sum()


class RepresentativeBatchSampler(torch.utils.data.Sampler):
    """Batches in which every class appears, as the BONN reference code does
    (generate_batches(..., mode='representative') in MLPBayesian-Python).

    Class k contributes ceil(batch_size * n_k / N) samples to every batch (at least 1), drawn
    from a fresh permutation of its samples repeated as needed, so minority classes are
    repeated rather than missing. The epoch has ceil(N / batch_size) batches. Uses numpy's
    global RNG, which run_experiment seeds."""

    def __init__(self, labels, batch_size: int, checks_per_epoch: int = 1):
        # checks_per_epoch > 1: each iteration covers only 1/checks of a pass over the data,
        # so the trainer can validate several times per pass
        self.labels = np.asarray(labels).round().astype(int)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.n_batches = int(np.ceil(len(self.labels) / batch_size / checks_per_epoch))

    def __iter__(self):
        N = len(self.labels)
        pools = []
        for c in self.classes:
            idx = np.flatnonzero(self.labels == c)
            per = int(np.ceil(self.batch_size * len(idx) / N))
            pools.append((per, np.resize(np.random.permutation(idx), per * self.n_batches)))
        for b in range(self.n_batches):
            batch = np.concatenate([pool[b * per:(b + 1) * per] for per, pool in pools])
            yield np.random.permutation(batch).tolist()

    def __len__(self):
        return self.n_batches


# =========================
# Metrics helpers
# =========================

def remap_to_3_classes(labels: np.ndarray) -> np.ndarray:
    """{1,2} → 1,  {3} → 2,  {4,5} → 3."""
    result = np.empty_like(labels)
    result[np.isin(labels, [1, 2])] = 1
    result[labels == 3]             = 2
    result[np.isin(labels, [4, 5])] = 3
    return result


def compute_range_distribution(true: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    """Proportion of samples at each absolute error level (0 … num_classes-1)."""
    err = np.abs(pred - true)
    n   = max(len(true), 1)
    return {f"range_{r}_pct": float(np.sum(err == r) / n) for r in range(num_classes)}


def compute_classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_labels: List[int],
) -> dict:
    """Return balanced accuracy, QWK, Spearman r, and per-class precision/recall/F1."""
    bal_acc    = balanced_accuracy_score(true_labels, pred_labels)
    qwk        = cohen_kappa_score(true_labels, pred_labels, weights="quadratic",
                                   labels=class_labels)
    spearman_r, _ = spearmanr(true_labels, pred_labels)
    precision  = precision_score(true_labels, pred_labels, labels=class_labels,
                                 average=None, zero_division=0)
    recall     = recall_score(true_labels, pred_labels, labels=class_labels,
                              average=None, zero_division=0)
    f1         = f1_score(true_labels, pred_labels, labels=class_labels,
                          average=None, zero_division=0)
    row = {
        "balanced_accuracy": float(bal_acc),
        "qwk":               float(qwk),
        "spearman_r":        float(spearman_r),
    }
    for i, c in enumerate(class_labels):
        row[f"precision_c{c}"] = float(precision[i])
        row[f"recall_c{c}"]    = float(recall[i])
        row[f"f1_c{c}"]        = float(f1[i])
    return row


# =========================
# Training / evaluation
# =========================

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_weights: bool = False,
    augmented_classes: Optional[List[int]] = None,
    bonn_thresholds: Optional[nn.Module] = None,
    bonn_class_counts: Optional[dict] = None,
    penalty_fn=None,
) -> Tuple[float, dict, dict]:
    """Train for one epoch.

    penalty_fn: optional callable returning a scalar tensor added to the loss before the
    backward pass (e.g. the L2-SP penalty). The reported loss excludes it, so it stays
    comparable with runs trained without a penalty.

    Returns:
        loss           : mean training loss
        class_counts   : {class_id: n_samples_seen}
        augmented_counts: {class_id: n_augmented_samples_seen}
    """
    model.train()
    running_loss     = 0.0
    total            = 0
    class_counts:     dict = {}
    augmented_counts: dict = {}
    aug_set = set(augmented_classes) if augmented_classes else set()

    for images, ratings, weights in loader:
        images  = images.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)
        if use_weights:
            weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images).squeeze(1)
        if bonn_thresholds is not None:
            loss_vec = criterion(preds, ratings, bonn_thresholds(), bonn_class_counts)
        else:
            loss_vec = criterion(preds, ratings)
        if loss_vec.ndim > 0:
            loss = (loss_vec * weights).mean() if use_weights else loss_vec.mean()
        else:
            loss = loss_vec

        (loss + penalty_fn() if penalty_fn is not None else loss).backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total        += images.size(0)

        # Track per-class and augmented counts
        for c in ratings.round().long().cpu().tolist():
            class_counts[c]  = class_counts.get(c, 0) + 1
            if c in aug_set:
                augmented_counts[c] = augmented_counts.get(c, 0) + 1

    return running_loss / max(total, 1), class_counts, augmented_counts


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_weights: bool = False,
    bonn_thresholds: Optional[nn.Module] = None,
    bonn_class_counts: Optional[dict] = None,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_mae  = 0.0
    total = 0

    for images, ratings, weights in loader:
        images  = images.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)
        preds = model(images).squeeze(1)

        if bonn_thresholds is not None:
            loss_vec = criterion(preds, ratings, bonn_thresholds(), bonn_class_counts)
        else:
            loss_vec = criterion(preds, ratings)
        if loss_vec.ndim > 0:
            if use_weights:
                weights = weights.to(device, non_blocking=True)
                loss = (loss_vec * weights).mean()
            else:
                loss = loss_vec.mean()
        else:
            loss = loss_vec

        # With BONN, z is not in [1,5] scale — use threshold-based predictions for MAE
        if bonn_thresholds is not None:
            mae = torch.abs(bonn_thresholds.predict(preds).float() - ratings).mean()
        else:
            mae = torch.abs(torch.clamp(preds, 1.0, 5.0) - ratings).mean()

        running_loss += loss.item() * images.size(0)
        running_mae  += mae.item()  * images.size(0)
        total        += images.size(0)

    return running_loss / max(total, 1), running_mae / max(total, 1)


@torch.no_grad()
def evaluate_dataset(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    num_classes: int = 5,
    bonn_thresholds: Optional[nn.Module] = None,
    bonn_class_counts: Optional[dict] = None,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a dataset.

    Returns:
      metrics_row : dict  — loss, MAE, and error-range distribution
      cm          : np.ndarray [C, C]  — confusion matrix (rows=true, cols=pred)
      all_true    : integer class labels
      all_pred    : predicted integer labels (rounded + clamped)
    """
    model.eval()

    total            = 0
    sum_loss         = 0.0
    sum_mae          = 0.0
    range_counts     = torch.zeros(num_classes, dtype=torch.long)
    cm               = torch.zeros((num_classes, num_classes), dtype=torch.long)
    all_true_list: List[np.ndarray] = []
    all_pred_list: List[np.ndarray] = []

    for images, ratings, _weights in loader:
        images  = images.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)

        z             = model(images).squeeze(1)
        preds_clamped = z.clamp(1.0, float(num_classes))

        if bonn_thresholds is not None:
            loss_c   = loss_fn(z, ratings, bonn_thresholds(), bonn_class_counts)
            pred_int = bonn_thresholds.predict(z)
            mae_c    = torch.abs(pred_int.float() - ratings).mean()
        else:
            loss_c   = loss_fn(preds_clamped, ratings)
            pred_int = torch.round(preds_clamped).to(torch.long).clamp(1, num_classes)
            mae_c    = torch.abs(preds_clamped - ratings).mean()

        true_int = torch.round(ratings).to(torch.long).clamp(1, num_classes)

        err_range = (pred_int - true_int).abs().clamp(0, num_classes - 1)
        range_counts += torch.bincount(err_range, minlength=num_classes).cpu()

        t = (true_int - 1).view(-1)
        p = (pred_int - 1).view(-1)
        cm += (torch.bincount(t * num_classes + p, minlength=num_classes * num_classes)
               .view(num_classes, num_classes).cpu())

        all_true_list.append(true_int.cpu().numpy())
        all_pred_list.append(pred_int.cpu().numpy())

        bsz      = images.size(0)
        total   += bsz
        sum_loss += loss_c.item() * bsz
        sum_mae  += mae_c.item()  * bsz

    range_prop = range_counts.float() / max(total, 1)

    metrics_row = {
        "dataset":      name,
        "loss_clamped": sum_loss / max(total, 1),
        "mae_clamped":  sum_mae  / max(total, 1),
    }
    for r in range(num_classes):
        metrics_row[f"range_{r}_pct"] = float(range_prop[r].item())

    if all_true_list:
        all_true = np.concatenate(all_true_list)
        all_pred = np.concatenate(all_pred_list)
    else:
        all_true = np.array([], dtype=np.int64)
        all_pred = np.array([], dtype=np.int64)

    return metrics_row, cm.numpy(), all_true, all_pred


def print_confusion_matrix(
    cm: np.ndarray,
    title: str,
    labels=None,
    logger: Optional[Logger] = None,
) -> None:
    if labels is None:
        labels = [1, 2, 3, 4, 5]
    log = logger.log if logger else print
    log("\n" + title)
    header = "true\\pred | " + " ".join([f"{l:>5}" for l in labels])
    log(header)
    log("-" * len(header))
    for i, l in enumerate(labels):
        row = " ".join([f"{cm[i, j]:>5d}" for j in range(len(labels))])
        log(f"{l:>9} | {row}")


# =========================
# Autoencoder
# =========================

def build_autoencoder_transforms(img_height: int, img_width: int) -> transforms.Compose:
    """Resize + ToTensor only (no ImageNet normalization) so decoder targets [0,1]."""
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])


def build_autoencoder_augmented_transforms(img_height: int, img_width: int) -> transforms.Compose:
    """Heavy augmentation for autoencoder training (no ImageNet normalization)."""
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3),
    ])


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder for road segment images.

    Computes output_padding per layer from *img_height* and *img_width* at init
    so that the decoder exactly recovers the input spatial dimensions.

    Default (202×54):
        Encoder: 202×54 → 101×27 → 51×14 → 26×7 → 13×4  (bottleneck: 256×13×4)
        Decoder mirrors encoder with ConvTranspose2d, Sigmoid output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        img_height: int = 202,
        img_width: int = 54,
    ):
        super().__init__()
        c = base_channels

        # Walk through encoder spatial sizes to compute decoder output_padding
        h, w = img_height, img_width
        enc_hw = [(h, w)]
        for _ in range(4):
            h = (h + 2 * 1 - 3) // 2 + 1   # Conv2d(k=3, s=2, p=1)
            w = (w + 2 * 1 - 3) // 2 + 1
            enc_hw.append((h, w))
        # enc_hw: [(202,54), (101,27), (51,14), (26,7), (13,4)]

        def _out_pad(src_idx: int) -> Tuple[int, int]:
            """output_padding to go from enc_hw[src_idx] back to enc_hw[src_idx-1]."""
            ih, iw = enc_hw[src_idx]
            th, tw = enc_hw[src_idx - 1]
            return (th - (2 * ih - 1), tw - (2 * iw - 1))

        # Encoder
        self.enc1 = self._enc_block(in_channels, c)
        self.enc2 = self._enc_block(c, c * 2)
        self.enc3 = self._enc_block(c * 2, c * 4)
        self.enc4 = self._enc_block(c * 4, c * 8)

        # Decoder
        self.dec4 = self._dec_block(c * 8, c * 4, _out_pad(4))
        self.dec3 = self._dec_block(c * 4, c * 2, _out_pad(3))
        self.dec2 = self._dec_block(c * 2, c,     _out_pad(2))
        op1 = _out_pad(1)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c, in_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=op1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _enc_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _dec_block(in_c: int, out_c: int, output_padding: Tuple[int, int]) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2,
                               padding=1, output_padding=output_padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x


class UNetAutoencoder(nn.Module):
    """U-Net style autoencoder with skip connections.

    Same 4-layer encoder as ConvAutoencoder, but the decoder concatenates
    encoder feature maps at each level, then reduces channels with a 1x1 conv.
    This preserves spatial detail through shortcuts while the bottleneck
    captures abstract features.

    Default (202x54):
        Encoder: 202x54 -> 101x27 -> 51x14 -> 26x7 -> 13x4
        Decoder mirrors encoder with skip connections from enc3, enc2, enc1.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        img_height: int = 202,
        img_width: int = 54,
    ):
        super().__init__()
        c = base_channels

        # Walk through encoder spatial sizes (same as ConvAutoencoder)
        h, w = img_height, img_width
        enc_hw = [(h, w)]
        for _ in range(4):
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1
            enc_hw.append((h, w))

        def _out_pad(src_idx: int) -> Tuple[int, int]:
            ih, iw = enc_hw[src_idx]
            th, tw = enc_hw[src_idx - 1]
            return (th - (2 * ih - 1), tw - (2 * iw - 1))

        # Encoder
        self.enc1 = ConvAutoencoder._enc_block(in_channels, c)
        self.enc2 = ConvAutoencoder._enc_block(c, c * 2)
        self.enc3 = ConvAutoencoder._enc_block(c * 2, c * 4)
        self.enc4 = ConvAutoencoder._enc_block(c * 4, c * 8)

        # Decoder with skip connections
        self.dec4 = ConvAutoencoder._dec_block(c * 8, c * 4, _out_pad(4))
        self.skip3 = nn.Conv2d(c * 4 * 2, c * 4, kernel_size=1)  # concat -> reduce

        self.dec3 = ConvAutoencoder._dec_block(c * 4, c * 2, _out_pad(3))
        self.skip2 = nn.Conv2d(c * 2 * 2, c * 2, kernel_size=1)

        self.dec2 = ConvAutoencoder._dec_block(c * 2, c, _out_pad(2))
        self.skip1 = nn.Conv2d(c * 2, c, kernel_size=1)

        op1 = _out_pad(1)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c, in_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=op1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4)
        d4 = self.skip3(torch.cat([d4, e3], dim=1))

        d3 = self.dec3(d4)
        d3 = self.skip2(torch.cat([d3, e2], dim=1))

        d2 = self.dec2(d3)
        d2 = self.skip1(torch.cat([d2, e1], dim=1))

        return self.dec1(d2)


def train_autoencoder_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train autoencoder for one epoch using MSE reconstruction loss.

    Returns mean reconstruction MSE.
    """
    model.train()
    running_loss = 0.0
    total = 0
    criterion = nn.MSELoss()

    for images, _ratings, _weights in loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / max(total, 1)


@torch.no_grad()
def validate_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate autoencoder. Returns mean MSE."""
    model.eval()
    running_loss = 0.0
    total = 0
    criterion = nn.MSELoss()

    for images, _ratings, _weights in loader:
        images = images.to(device, non_blocking=True)
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / max(total, 1)


@torch.no_grad()
def classify_by_reconstruction(
    autoencoders: dict,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify each sample by the autoencoder with minimum reconstruction MSE.

    Args:
        autoencoders: {class_id: model} where class_id is 1-indexed.
        loader:       DataLoader yielding (images, ratings, weights).
        device:       torch device.
        num_classes:  number of classes (default 5).

    Returns:
        all_true : int array [N] — ground truth class labels (1-indexed)
        all_pred : int array [N] — predicted class labels (1-indexed)
        mse_matrix : float array [N, num_classes] — per-sample MSE for each autoencoder
    """
    for ae in autoencoders.values():
        ae.eval()

    all_true_list: List[np.ndarray] = []
    all_pred_list: List[np.ndarray] = []
    all_mse_list:  List[np.ndarray] = []

    class_ids = sorted(autoencoders.keys())

    for images, ratings, _weights in loader:
        images = images.to(device, non_blocking=True)
        B = images.size(0)

        mse_per_class = torch.zeros(B, num_classes, device=device)
        for i, k in enumerate(class_ids):
            reconstructed = autoencoders[k](images)
            # Per-sample MSE (mean over C, H, W)
            mse_per_class[:, i] = ((images - reconstructed) ** 2).mean(dim=(1, 2, 3))

        pred_idx = mse_per_class.argmin(dim=1)
        pred_labels = torch.tensor([class_ids[j] for j in pred_idx.cpu().tolist()],
                                   dtype=torch.long)
        true_labels = ratings.round().long().clamp(1, num_classes)

        all_true_list.append(true_labels.cpu().numpy())
        all_pred_list.append(pred_labels.numpy())
        all_mse_list.append(mse_per_class.cpu().numpy())

    all_true = np.concatenate(all_true_list)
    all_pred = np.concatenate(all_pred_list)
    mse_matrix = np.concatenate(all_mse_list, axis=0)
    return all_true, all_pred, mse_matrix


@torch.no_grad()
def visualise_reconstructions(
    autoencoders: dict,
    loader: DataLoader,
    device: torch.device,
    output_path,
    num_classes: int = 5,
    samples_per_class: int = 4,
) -> None:
    """Save grid images showing original vs reconstructed for each class.

    For each ground-truth class, produces a PNG with rows = samples and
    columns = [Original, AE_1, AE_2, ..., AE_K].  The column matching
    the sample's true class is highlighted with a green border.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    for ae in autoencoders.values():
        ae.eval()

    class_ids = sorted(autoencoders.keys())
    output_path = Path(output_path)

    # Collect samples per class
    collected: dict = {k: [] for k in range(1, num_classes + 1)}
    for images, ratings, _weights in loader:
        for i in range(images.size(0)):
            cls = int(ratings[i].round().item())
            if 1 <= cls <= num_classes and len(collected[cls]) < samples_per_class:
                collected[cls].append(images[i])
        if all(len(v) >= samples_per_class for v in collected.values()):
            break

    n_cols = 1 + len(class_ids)  # original + one per autoencoder
    for cls in range(1, num_classes + 1):
        samples = collected[cls]
        if not samples:
            continue
        n_rows = len(samples)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for ri, img_tensor in enumerate(samples):
            img_dev = img_tensor.unsqueeze(0).to(device)
            orig_np = img_tensor.permute(1, 2, 0).cpu().numpy()

            axes[ri, 0].imshow(orig_np)
            axes[ri, 0].set_ylabel(f"Sample {ri+1}", fontsize=9)
            if ri == 0:
                axes[ri, 0].set_title("Original", fontsize=10)
            axes[ri, 0].set_xticks([])
            axes[ri, 0].set_yticks([])

            for ci, k in enumerate(class_ids):
                recon = autoencoders[k](img_dev)
                recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
                ax = axes[ri, ci + 1]
                ax.imshow(recon_np)
                if ri == 0:
                    ax.set_title(f"AE {k}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

                # Highlight the matching autoencoder
                if k == cls:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("green")
                        spine.set_linewidth(3)

        fig.suptitle(f"Class {cls} reconstructions", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path / f"reconstructions_class_{cls}.png", dpi=120)
        plt.close(fig)


# =========================
# Calibration
# =========================

class GaussianCalibrator:
    """Gaussian boundary calibration for regression-to-probability conversion.

    P(rating=k | ŷ) = Φ((b_k − ŷ) / σ) − Φ((b_{k−1} − ŷ) / σ)

    Boundaries: 0.5, 1.5, 2.5, 3.5, 4.5, 5.5  (midpoints between integer ratings,
    with 0.5 and 5.5 as outer edges).
    σ is estimated from the std dev of (y_true − ŷ) on validation data.
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.boundaries = np.array([k + 0.5 for k in range(num_classes + 1)])  # [0.5, 1.5, ..., 5.5]
        self.sigma: Optional[float] = None

    def fit(self, y_true: np.ndarray, y_hat: np.ndarray) -> "GaussianCalibrator":
        """Estimate σ from validation residuals."""
        residuals = y_true.astype(float) - y_hat.astype(float)
        self.sigma = float(np.std(residuals))
        if self.sigma < 1e-8:
            self.sigma = 1e-8
        return self

    def predict_proba(self, y_hat: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities [N, num_classes]."""
        from scipy.stats import norm
        y = np.asarray(y_hat, dtype=float).reshape(-1, 1)        # [N, 1]
        b = self.boundaries.reshape(1, -1)                        # [1, K+1]
        cdf_vals = norm.cdf((b - y) / self.sigma)                 # [N, K+1]
        probs = np.diff(cdf_vals, axis=1)                         # [N, K]
        probs = np.clip(probs, 0.0, None)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-12)
        return probs

    def predict(self, y_hat: np.ndarray) -> np.ndarray:
        """Return predicted class labels (1-indexed)."""
        probs = self.predict_proba(y_hat)
        return probs.argmax(axis=1) + 1

    def get_params(self) -> dict:
        return {"sigma": self.sigma, "boundaries": self.boundaries.tolist()}


class OrdinalLogisticCalibrator:
    """Proportional-odds ordinal logistic regression on the scalar regression output.

    P(Y ≤ k | ŷ) = sigmoid(α_k − β · ŷ)  for k = 1, ..., K-1
    P(Y = k) = P(Y ≤ k) − P(Y ≤ k−1)

    Fit α_k and β on validation data via maximum likelihood (scipy.optimize).
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.alphas: Optional[np.ndarray] = None   # [K-1] ordered thresholds
        self.beta:   Optional[float] = None

    def fit(self, y_true: np.ndarray, y_hat: np.ndarray) -> "OrdinalLogisticCalibrator":
        """Fit proportional-odds model on validation data."""
        from scipy.optimize import minimize

        K = self.num_classes
        y_int = np.round(y_true).astype(int).clip(1, K)
        x = y_hat.astype(float)

        # Initial guess: evenly spaced thresholds, positive slope
        alpha_init = np.linspace(-1.0, 1.0, K - 1)
        beta_init = 1.0

        def _neg_log_likelihood(params):
            # params: [alpha_1, ..., alpha_{K-1}, beta]
            # alphas must be ordered; enforce via cumulative softplus
            raw_alphas = params[:K - 1]
            beta = params[K - 1]
            alphas = np.empty(K - 1)
            alphas[0] = raw_alphas[0]
            for i in range(1, K - 1):
                alphas[i] = alphas[i - 1] + np.log1p(np.exp(raw_alphas[i]))

            def _sigmoid(z):
                return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

            # Cumulative probabilities: P(Y <= k) for k=1..K-1
            cum_probs = _sigmoid(alphas[np.newaxis, :] - beta * x[:, np.newaxis])  # [N, K-1]
            # Pad with 0 on left, 1 on right
            cum_full = np.concatenate([
                np.zeros((len(x), 1)),
                cum_probs,
                np.ones((len(x), 1)),
            ], axis=1)  # [N, K+1]

            # P(Y = k) for each sample
            probs = np.diff(cum_full, axis=1)  # [N, K]
            probs = np.clip(probs, 1e-12, None)

            # Pick the probability of the true class
            idx = y_int - 1  # 0-indexed
            log_lik = np.log(probs[np.arange(len(x)), idx])
            return -log_lik.sum()

        # Convert initial alphas to raw parameterisation
        raw_alpha_init = np.zeros(K - 1)
        raw_alpha_init[0] = alpha_init[0]
        for i in range(1, K - 1):
            delta = alpha_init[i] - alpha_init[i - 1]
            raw_alpha_init[i] = np.log(np.exp(max(delta, 1e-4)) - 1.0)

        x0 = np.concatenate([raw_alpha_init, [beta_init]])
        result = minimize(_neg_log_likelihood, x0, method="L-BFGS-B",
                          options={"maxiter": 1000})

        # Extract fitted parameters
        raw = result.x
        self.alphas = np.empty(K - 1)
        self.alphas[0] = raw[0]
        for i in range(1, K - 1):
            self.alphas[i] = self.alphas[i - 1] + np.log1p(np.exp(raw[i]))
        self.beta = float(raw[K - 1])
        return self

    def predict_proba(self, y_hat: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities [N, num_classes]."""
        x = np.asarray(y_hat, dtype=float)

        def _sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

        cum_probs = _sigmoid(self.alphas[np.newaxis, :] - self.beta * x[:, np.newaxis])
        cum_full = np.concatenate([
            np.zeros((len(x), 1)),
            cum_probs,
            np.ones((len(x), 1)),
        ], axis=1)
        probs = np.diff(cum_full, axis=1)
        probs = np.clip(probs, 0.0, None)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(row_sums, 1e-12)
        return probs

    def predict(self, y_hat: np.ndarray) -> np.ndarray:
        """Return predicted class labels (1-indexed)."""
        return self.predict_proba(y_hat).argmax(axis=1) + 1

    def get_params(self) -> dict:
        return {"alphas": self.alphas.tolist(), "beta": self.beta}


class TemperatureCalibrator:
    """Temperature-scaled softmax calibration for autoencoder MSE scores.

    P(class=k | x) = softmax(−MSE_k / T)

    Temperature T is fit on the validation set by minimising negative log-likelihood.
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.temperature: Optional[float] = None

    def fit(self, y_true: np.ndarray, mse_matrix: np.ndarray) -> "TemperatureCalibrator":
        """Fit temperature on validation MSE matrix [N, K]."""
        from scipy.optimize import minimize_scalar

        y_int = np.round(y_true).astype(int).clip(1, self.num_classes)
        idx = y_int - 1  # 0-indexed

        def _nll(log_T):
            T = np.exp(log_T)
            logits = -mse_matrix / T
            logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            return -log_probs[np.arange(len(idx)), idx].sum()

        result = minimize_scalar(_nll, bounds=(-10, 10), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def predict_proba(self, mse_matrix: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities [N, num_classes]."""
        logits = -mse_matrix / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, mse_matrix: np.ndarray) -> np.ndarray:
        """Return predicted class labels (1-indexed)."""
        return self.predict_proba(mse_matrix).argmax(axis=1) + 1

    def get_params(self) -> dict:
        return {"temperature": self.temperature}
