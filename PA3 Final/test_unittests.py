"""Synthetic, CPU-only tests for PA3's CNN experiment API."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import student_code as sc


def test_load_imagenet100_rejects_invalid_arguments() -> None:
  with pytest.raises(ValueError, match="split"):
    sc.load_imagenet100("test")
  with pytest.raises(ValueError, match="image_size"):
    sc.load_imagenet100("train", image_size=0)


def test_make_loaders_loads_both_splits_and_keeps_validation_order(monkeypatch) -> None:
  train_data = TensorDataset(torch.arange(6).view(-1, 1).float(), torch.arange(6))
  validation_data = TensorDataset(torch.arange(4).view(-1, 1).float(), torch.arange(4))
  calls: list[tuple[str, int, object]] = []

  def fake_load(split, image_size, cache_dir):
    calls.append((split, image_size, cache_dir))
    return train_data if split == "train" else validation_data

  monkeypatch.setattr(sc, "load_imagenet100", fake_load)
  train_loader, validation_loader = sc.make_imagenet100_loaders(
    batch_size=2, image_size=17, seed=4, cache_dir="cache"
  )

  assert calls == [("train", 17, "cache"), ("validation", 17, "cache")]
  assert torch.equal(torch.cat([labels for _, labels in validation_loader]), torch.arange(4))
  assert sorted(torch.cat([labels for _, labels in train_loader]).tolist()) == list(range(6))


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"num_workers": -1}])
def test_make_loaders_rejects_invalid_loader_options(kwargs: dict[str, int]) -> None:
  with pytest.raises(ValueError):
    sc.make_imagenet100_loaders(**kwargs)


def test_count_trainable_parameters_ignores_frozen_parameters() -> None:
  model = nn.Linear(3, 2)
  model.bias.requires_grad_(False)
  assert sc.count_trainable_parameters(model) == 6


def test_cnn_returns_unnormalized_logits_at_requested_class_count() -> None:
  model = sc.CNNClassifier(channels=[4, 8], num_classes=3)
  logits = model(torch.randn(2, 3, 16, 16))

  assert logits.shape == (2, 3)
  assert not torch.allclose(logits.softmax(dim=1), logits)


def test_cnn_registers_a_conv_batchnorm_and_pool_per_width() -> None:
  model = sc.CNNClassifier(channels=[4, 7, 9], num_classes=3)

  assert len([m for m in model.modules() if isinstance(m, nn.Conv2d)]) == 3
  assert len([m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]) == 3
  assert len([m for m in model.modules() if isinstance(m, nn.MaxPool2d)]) == 3
  assert isinstance(model.classifier[-1], nn.Linear)
  assert model.classifier[-1].out_features == 3


@pytest.mark.parametrize(
  "kwargs",
  [
    {"channels": []}, {"channels": [4, 0]}, {"num_classes": 1}, {"dropout": 1.0},
    {"optimizer_name": "rmsprop"}, {"learning_rate": 0.0}, {"weight_decay": -0.1},
  ],
)
def test_cnn_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
  with pytest.raises(ValueError):
    sc.CNNClassifier(**kwargs)


def test_loss_matches_cross_entropy_and_training_step_retains_gradients() -> None:
  model = sc.CNNClassifier(channels=[2], num_classes=2)
  logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
  assert model.loss(logits, torch.tensor([0, 1])).item() == pytest.approx(0.126928, abs=1e-5)

  loss, correct, total = model.training_step((torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 0])))
  assert loss.requires_grad
  assert 0 <= correct <= total == 3


@pytest.mark.parametrize("name, optimizer_type", [("sgd", torch.optim.SGD), ("adamw", torch.optim.AdamW)])
def test_configure_optimizers_uses_requested_optimizer(name: str, optimizer_type: type) -> None:
  model = sc.CNNClassifier(channels=[2], optimizer_name=name, learning_rate=0.02, weight_decay=0.03)
  optimizer = model.configure_optimizers()

  assert isinstance(optimizer, optimizer_type)
  assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
  assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.03)
  assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
    id(p) for p in model.parameters()
  }


def test_evaluate_weights_batches_disables_gradients_and_restores_mode() -> None:
  class FixedModel(sc.CNNClassifier):
    def __init__(self) -> None:
      nn.Module.__init__(self)
      self.seen_grad_enabled: list[bool] = []

    def validation_step(self, batch):
      self.seen_grad_enabled.append(torch.is_grad_enabled())
      _, labels = batch
      logits = torch.tensor([[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]])[: len(labels)]
      return nn.functional.cross_entropy(logits, labels), (logits.argmax(1) == labels).sum().item(), len(labels)

  model = FixedModel()
  loader = DataLoader(TensorDataset(torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 1])), batch_size=2)
  model.train()
  metrics = sc.evaluate(model, loader)

  expected_logits = torch.tensor([[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]])
  assert metrics["loss"] == pytest.approx(nn.functional.cross_entropy(expected_logits, torch.tensor([0, 1, 1])).item())
  assert metrics["accuracy"] == pytest.approx(2 / 3)
  assert model.seen_grad_enabled == [False, False]
  assert model.training


def test_batchnorm_does_not_update_during_evaluation() -> None:
  model = sc.CNNClassifier(channels=[3], num_classes=2)
  batchnorm = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
  before = batchnorm.running_mean.clone()
  sc.evaluate(model, DataLoader(TensorDataset(torch.randn(4, 3, 8, 8), torch.zeros(4, dtype=torch.long)), batch_size=2))
  assert torch.equal(batchnorm.running_mean, before)


def test_train_model_overfits_a_tiny_image_problem_and_records_history() -> None:
  X = torch.cat([torch.zeros(8, 3, 8, 8), torch.ones(8, 3, 8, 8)])
  y = torch.tensor([0] * 8 + [1] * 8)
  loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True)
  model = sc.CNNClassifier(channels=[4], num_classes=2, optimizer_name="adamw", learning_rate=0.03)

  history = sc.train_model(model, loader, loader, n_epochs=12)

  assert set(history) == {"train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "epoch_seconds"}
  assert all(len(values) == 12 for values in history.values())
  assert history["validation_accuracy"][-1] > 0.95
  assert all(seconds >= 0 for seconds in history["epoch_seconds"])


def test_train_model_rejects_bad_epoch_count_and_empty_train_loader() -> None:
  loader = DataLoader(TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long)), batch_size=2)
  model = sc.CNNClassifier(channels=[2], num_classes=2)
  with pytest.raises(ValueError, match="epochs"):
    sc.train_model(model, loader, loader, n_epochs=0)
  with pytest.raises(ValueError, match="empty"):
    sc.train_model(model, loader, loader, n_epochs=1)
