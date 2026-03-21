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
    gradient_backend: str = "auto"
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    restore_best: bool = True
    initialization_strategy: str = "current"
    initialization_noise_scale: float = 5e-2
    num_restarts: int = 1
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.finite_difference_eps <= 0.0:
            raise ValueError("finite_difference_eps must be positive")
        if self.gradient_backend not in {"auto", "exact", "finite_difference"}:
            raise ValueError(
                "gradient_backend must be 'auto', 'exact', or 'finite_difference'"
            )
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if self.initialization_strategy not in {"current", "noisy_current"}:
            raise ValueError("initialization_strategy must be 'current' or 'noisy_current'")
        if self.initialization_noise_scale < 0.0:
            raise ValueError("initialization_noise_scale must be non-negative")
        if self.num_restarts < 1:
            raise ValueError("num_restarts must be at least 1")


class Trainer:
    """Simple optimizer loop for small parameter counts."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def fit(self, model: object, dataset: DatasetSplit | DatasetBundle) -> dict[str, object]:
        split = self._coerce_split(dataset)
        base_parameters = model.get_parameters()
        rng = np.random.default_rng(self.config.random_seed)

        restart_histories: list[dict[str, object]] = []
        best_restart = 0
        best_history: dict[str, object] | None = None

        for restart_index in range(self.config.num_restarts):
            initial_parameters = self._initialize_parameters(base_parameters, rng)
            model.set_parameters(initial_parameters)
            history = self._fit_once(model, split, initial_parameters)
            restart_histories.append(history)

            if best_history is None or float(history["best_loss"]) < float(best_history["best_loss"]):
                best_history = history
                best_restart = restart_index

        assert best_history is not None

        if self.config.restore_best:
            model.set_parameters(np.asarray(best_history["best_parameters"], dtype=np.float64))

        result = dict(best_history)
        result["best_restart"] = best_restart
        result["restart_histories"] = restart_histories
        return result

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

    def gradient(
        self,
        model: object,
        dataset: DatasetSplit | DatasetBundle,
        *,
        parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        split = self._coerce_split(dataset)
        parameter_array = model.get_parameters() if parameters is None else np.asarray(parameters, dtype=np.float64)
        labels = np.asarray(split.labels, dtype=np.float64)
        return self._loss_gradient(model, split.states, labels, parameter_array)

    def _coerce_split(self, dataset: DatasetSplit | DatasetBundle) -> DatasetSplit:
        return dataset.train if isinstance(dataset, DatasetBundle) else dataset

    def _initialize_parameters(
        self,
        base_parameters: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.config.initialization_strategy == "current":
            return base_parameters.copy()

        noise = rng.normal(
            loc=0.0,
            scale=self.config.initialization_noise_scale,
            size=base_parameters.shape,
        )
        return np.asarray(base_parameters + noise, dtype=np.float64)

    def _fit_once(
        self,
        model: object,
        split: DatasetSplit,
        initial_parameters: np.ndarray,
    ) -> dict[str, object]:
        states = split.states
        labels = np.asarray(split.labels, dtype=np.float64)
        parameters = np.asarray(initial_parameters, dtype=np.float64).copy()

        first_metrics = self.evaluate(model, split, parameters=parameters)

        history: dict[str, object] = {
            "loss": [first_metrics["loss"]],
            "accuracy": [first_metrics["accuracy"]],
            "best_loss": first_metrics["loss"],
            "best_accuracy": first_metrics["accuracy"],
            "best_parameters": parameters.copy(),
            "initial_parameters": parameters.copy(),
        }

        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)

        for epoch in range(1, self.config.epochs + 1):
            gradient = self._loss_gradient(model, states, labels, parameters)

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

        history["final_parameters"] = parameters.copy()
        return history

    def _loss_gradient(
        self,
        model: object,
        states: np.ndarray,
        labels: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        if self.config.gradient_backend in {"auto", "exact"} and hasattr(model, "loss_gradient"):
            try:
                return np.asarray(
                    model.loss_gradient(
                        states,
                        labels,
                        parameters=parameters,
                        finite_difference_eps=self.config.finite_difference_eps,
                    ),
                    dtype=np.float64,
                )
            except NotImplementedError:
                if self.config.gradient_backend == "exact":
                    raise

        if self.config.gradient_backend == "exact":
            raise ValueError("Exact gradients are not available for this model")

        return self._finite_difference_gradient(model, states, labels, parameters)

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
