"""Tools for designing CNN image classifiers on ImageNet-100.

The loader and training loop are deliberately supplied.  The CNN class uses a
small baseline architecture, which you should treat as a starting point for
your own experiments rather than as a required final design.
"""

import random
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import batch_norm, max_pool2d, cross_entropy
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 128
NUM_CLASSES = 100
DATASET_REPOSITORY = "clane9/imagenet-100"
# Pin the revision so that the course task cannot change if the Hub's main
# branch later changes. This commit changes the dataset card only.
DATASET_REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"


def set_seed(seed: int) -> None:  # redaction: keep
  """Seed Python and PyTorch so that one experimental run is reproducible."""
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class _TransformedImageDataset(Dataset):
  """Adapt a Hugging Face image dataset to PyTorch's ``(image, label)`` API."""

  def __init__(self, dataset: object, transform: object) -> None:  # redaction: keep
    self.dataset = dataset
    self.transform = transform

  def __len__(self) -> int:  # redaction: keep
    return len(self.dataset)  # type: ignore[arg-type]

  def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:  # redaction: keep
    example = self.dataset[index]  # type: ignore[index]
    return self.transform(example["image"].convert("RGB")), int(example["label"])


# redaction: keep
def load_imagenet100(
  split: str,
  image_size: int = IMAGE_SIZE,
  cache_dir: str | Path | None = None,
) -> Dataset:
  """Download/load the pinned ImageNet-100 split from Hugging Face.

  Load the course's pinned ``clane9/imagenet-100`` revision through the
  established ``datasets`` package. ``split`` is ``"train"`` or
  ``"validation"``. Training images use random resized crops and horizontal
  flips; validation images use deterministic resize and center crop. Set
  ``cache_dir`` to choose where Hugging Face stores its reusable local cache.
  Returned items are normalized ``float32`` tensors with shape
  ``(3, image_size, image_size)`` and integer labels. Raise ``ValueError`` for
  an unknown split or non-positive image size.
  """
  from datasets import load_dataset
  from torchvision import transforms

  if split not in {"train", "validation"}:
    raise ValueError(f"unknown split: {split}")
  if image_size <= 0:
    raise ValueError("image_size must be positive")

  normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
  )
  if split == "train":
    transform = transforms.Compose([
      transforms.RandomResizedCrop(image_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ])
  else:
    resize_size = round(image_size * 256 / 224)
    transform = transforms.Compose([
      transforms.Resize(resize_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
      normalize,
    ])
  dataset = load_dataset(
    DATASET_REPOSITORY,
    split=split,
    revision=DATASET_REVISION,
    cache_dir=str(cache_dir) if cache_dir is not None else None,
  )
  return _TransformedImageDataset(dataset, transform)


# redaction: keep
def make_imagenet100_loaders(
  batch_size: int = 64,
  image_size: int = IMAGE_SIZE,
  seed: int = 0,
  num_workers: int = 0,
  cache_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
  """Return reproducible shuffled training and ordered validation loaders.

  Both loaders use the pinned Hugging Face dataset and have the requested
  ``batch_size``. Training is shuffled with a generator seeded by ``seed``;
  validation is not shuffled. ``num_workers`` and ``cache_dir`` are passed to
  the appropriate loader/data source. Raise ``ValueError`` if ``batch_size``
  is non-positive or ``num_workers`` is negative.
  """
  if batch_size <= 0:
    raise ValueError("batch_size must be positive")
  if num_workers < 0:
    raise ValueError("num_workers cannot be negative")
  train_data = load_imagenet100("train", image_size, cache_dir)
  validation_data = load_imagenet100("validation", image_size, cache_dir)
  generator = torch.Generator().manual_seed(seed)
  train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, generator=generator,
    num_workers=num_workers,
  )
  validation_loader = DataLoader(
    validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
  )
  return train_loader, validation_loader


def count_trainable_parameters(model: nn.Module) -> int:  # redaction: keep
  """Return the number of scalar parameters that will be optimized."""
  return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class CNNClassifier(nn.Module):
  """A configurable convolutional classifier with packaged training choices.

  The baseline has one Conv2d--BatchNorm2d--ReLU--MaxPool2d block per entry
  in ``channels``, followed by global average pooling, dropout, and a linear
  100-class output layer. ``channels=[32, 64, 128]`` is a sensible starting
  point, not a limit on the architectures you may investigate.
  """

  def __init__(
    self,
    channels: Sequence[int] = (32, 64, 128),
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.0,
    optimizer_name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
  ) -> None:
    """Build the baseline CNN and save its optimizer configuration.

    Every entry in ``channels`` is a positive convolution width and at least
    one entry is required. ``num_classes`` must exceed one, ``dropout`` must
    be in ``[0, 1)``, ``optimizer_name`` is ``"sgd"`` or ``"adamw"``, and
    learning rate and weight decay must be non-negative as appropriate.
    """
    super().__init__()
    if (not channels or
            any(width <= 0 for width in channels) or
            num_classes <= 1 or
            dropout < 0 or
            dropout >= 1 or
            optimizer_name not in ("sgd", "adamw") or
            learning_rate <= 0 or
            weight_decay < 0):
      raise ValueError("Invalid model configuration")

    self.optimizer_name = optimizer_name
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

    layers = []
    input_channels = 3

    for output_channels in channels:
      layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
      layers.append(nn.BatchNorm2d(output_channels))
      layers.append(nn.ReLU())
      layers.append(nn.MaxPool2d(kernel_size=2))
      input_channels = output_channels

    self.features = nn.Sequential(*layers)
    self.classifier = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(start_dim=1),
      nn.Dropout(p=dropout),
      nn.Linear(input_channels, num_classes),
    )

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    """Return unnormalized class logits with shape ``(N, num_classes)``."""
    features = self.features(X)
    logits = self.classifier(features)
    return logits

  def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return mean cross-entropy loss from logits and integer class labels."""
    return nn.functional.cross_entropy(logits, labels)

  def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, int, int]:
    """Return differentiable loss, correct-prediction count, and example count."""
    images, labels = batch
    logits = self(images)
    loss = self.loss(logits, labels)

    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    return loss, correct, total

  def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, int, int]:
    """Return validation metrics for one batch; the caller controls eval mode."""
    return self.training_step(batch)

  def configure_optimizers(self) -> torch.optim.Optimizer:
    """Create this model's SGD or AdamW optimizer over all trainable parameters."""
    if self.optimizer_name == "sgd":
      return torch.optim.SGD(self.parameters(),
                 lr = self.learning_rate,
                 weight_decay = self.weight_decay)
    else:
      return torch.optim.AdamW(self.parameters(),
                 lr = self.learning_rate,
                 weight_decay = self.weight_decay)

  def configure_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
    """Return an optional epoch-level scheduler; the baseline uses no scheduler."""
    return None


# redaction: keep
def _move_batch(
  batch: tuple[torch.Tensor, torch.Tensor], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Move a data-loader batch to ``device``. Kept separate for a readable loop."""
  X, labels = batch
  return X.to(device), labels.to(device)


# redaction: keep
def evaluate(
  model: CNNClassifier, data_loader: DataLoader, device: torch.device | str = "cpu"
) -> dict[str, float]:
  """Evaluate all examples without gradients and restore the previous mode.

  Return example-weighted ``{"loss": ..., "accuracy": ...}``. Raise
  ``ValueError`` for an empty loader.
  """
  was_training = model.training
  model.eval()
  loss_total = 0.0
  correct_total = 0
  example_total = 0
  with torch.no_grad():
    for batch in data_loader:
      loss, correct, total = model.validation_step(_move_batch(batch, device))
      loss_total += loss.item() * total
      correct_total += correct
      example_total += total
  if was_training:
    model.train()
  if example_total == 0:
    raise ValueError("cannot evaluate an empty data loader")
  return {"loss": loss_total / example_total, "accuracy": correct_total / example_total}


# redaction: keep
def train_model(
  model: CNNClassifier,
  train_loader: DataLoader,
  validation_loader: DataLoader,
  n_epochs: int,
  device: torch.device | str = "cpu",
) -> dict[str, list[float]]:
  """Train for ``n_epochs`` and record sample-weighted epoch metrics.

  Move ``model`` to ``device``, use its packaged optimizer and optional
  epoch-level scheduler, and return lists named ``train_loss``,
  ``train_accuracy``, ``validation_loss``, ``validation_accuracy``, and
  ``epoch_seconds``. Raise ``ValueError`` for non-positive epochs or an empty
  training loader.
  """
  if n_epochs <= 0:
    raise ValueError("n_epochs must be positive")
  model.to(device)
  optimizer = model.configure_optimizers()
  scheduler = model.configure_scheduler(optimizer)
  history = {
    "train_loss": [], "train_accuracy": [], "validation_loss": [],
    "validation_accuracy": [], "epoch_seconds": [],
  }
  for _ in range(n_epochs):
    epoch_start = time.perf_counter()
    model.train()
    loss_total = 0.0
    correct_total = 0
    example_total = 0
    for batch in train_loader:
      optimizer.zero_grad()
      loss, correct, total = model.training_step(_move_batch(batch, device))
      loss.backward()
      optimizer.step()
      loss_total += loss.item() * total
      correct_total += correct
      example_total += total
    if example_total == 0:
      raise ValueError("cannot train on an empty data loader")
    validation_metrics = evaluate(model, validation_loader, device)
    history["train_loss"].append(loss_total / example_total)
    history["train_accuracy"].append(correct_total / example_total)
    history["validation_loss"].append(validation_metrics["loss"])
    history["validation_accuracy"].append(validation_metrics["accuracy"])
    history["epoch_seconds"].append(time.perf_counter() - epoch_start)
    if scheduler is not None:
      scheduler.step()
  return history
