"""Training support for small EQNN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqnn.datasets.heisenberg import DatasetBundle, DatasetSplit


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 5e-2
    finite_difference_eps: float = 1e-3
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    restore_best: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.finite_difference_eps <= 0.0:
            raise ValueError("finite_difference_eps must be positive")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")


class Trainer:
    """Simple optimizer loop for small parameter counts."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def fit(self, model: object, dataset: DatasetSplit | DatasetBundle) -> dict[str, object]:
        split = dataset.train if isinstance(dataset, DatasetBundle) else dataset
        states = split.states
        labels = np.asarray(split.labels, dtype=np.float64)

        parameters = model.get_parameters()
        first_metrics = self.evaluate(model, split, parameters=parameters)

        history: dict[str, object] = {
            "loss": [first_metrics["loss"]],
            "accuracy": [first_metrics["accuracy"]],
            "best_loss": first_metrics["loss"],
            "best_accuracy": first_metrics["accuracy"],
            "best_parameters": parameters.copy(),
        }

        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)

        for epoch in range(1, self.config.epochs + 1):
            gradient = self._finite_difference_gradient(model, states, labels, parameters)

            if self.config.optimizer == "adam":
                first_moment = self.config.beta1 * first_moment + (1.0 - self.config.beta1) * gradient
                second_moment = self.config.beta2 * second_moment + (1.0 - self.config.beta2) * (gradient**2)
                first_unbiased = first_moment / (1.0 - self.config.beta1**epoch)
                second_unbiased = second_moment / (1.0 - self.config.beta2**epoch)
                parameters = parameters - self.config.learning_rate * first_unbiased / (
                    np.sqrt(second_unbiased) + self.config.epsilon
                )
            else:
                parameters = parameters - self.config.learning_rate * gradient

            model.set_parameters(parameters)
            metrics = self.evaluate(model, split, parameters=parameters)
            history["loss"].append(metrics["loss"])
            history["accuracy"].append(metrics["accuracy"])

            if metrics["loss"] < history["best_loss"]:
                history["best_loss"] = metrics["loss"]
                history["best_accuracy"] = metrics["accuracy"]
                history["best_parameters"] = parameters.copy()

        if self.config.restore_best:
            model.set_parameters(history["best_parameters"])

        return history

    def evaluate(
        self,
        model: object,
        dataset: DatasetSplit,
        *,
        parameters: np.ndarray | None = None,
    ) -> dict[str, float]:
        probabilities = model.predict_batch(dataset.states, parameters=parameters)
        predictions = (probabilities >= 0.5).astype(np.int64)
        labels = dataset.labels.astype(np.int64)
        accuracy = float(np.mean(predictions == labels))
        loss = model.binary_cross_entropy(dataset.states, dataset.labels, parameters=parameters)
        return {"loss": float(loss), "accuracy": accuracy}

    def _finite_difference_gradient(
        self,
        model: object,
        states: np.ndarray,
        labels: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        gradient = np.zeros_like(parameters)

        for index in range(parameters.size):
            offset = np.zeros_like(parameters)
            offset[index] = self.config.finite_difference_eps
            loss_plus = model.binary_cross_entropy(states, labels, parameters=parameters + offset)
            loss_minus = model.binary_cross_entropy(states, labels, parameters=parameters - offset)
            gradient[index] = (loss_plus - loss_minus) / (2.0 * self.config.finite_difference_eps)

        return gradient
