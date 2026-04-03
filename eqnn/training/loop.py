"""Training support for small EQNN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqnn.datasets.heisenberg import DatasetBundle, DatasetSplit
from eqnn.models.base import TrainableModel
from eqnn.utils.timing import RuntimeProfile, timed


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 5e-2
    loss: str = "bce"
    batch_size: int | None = None
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
    classification_threshold: float = 0.5
    threshold_update: str = "none"
    threshold_critical_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.loss not in {"bce", "mse"}:
            raise ValueError("loss must be 'bce' or 'mse'")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive when provided")
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
        if not 0.0 <= self.classification_threshold <= 1.0:
            raise ValueError("classification_threshold must lie in [0, 1]")
        if self.threshold_update not in {"none", "paper_nearest_critical"}:
            raise ValueError(
                "threshold_update must be 'none' or 'paper_nearest_critical'"
            )


class Trainer:
    """Simple optimizer loop for small parameter counts."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def fit(
        self,
        model: TrainableModel,
        dataset: DatasetSplit | DatasetBundle,
        *,
        profile: RuntimeProfile | None = None,
    ) -> dict[str, object]:
        split = self._coerce_split(dataset)
        base_parameters = model.get_parameters()
        rng = np.random.default_rng(self.config.random_seed)

        restart_histories: list[dict[str, object]] = []
        best_restart = 0
        best_history: dict[str, object] | None = None

        for restart_index in range(self.config.num_restarts):
            initial_parameters = self._initialize_parameters(base_parameters, rng)
            model.set_parameters(initial_parameters)
            self._initialize_model_threshold(model)
            history = self._fit_once(model, split, initial_parameters, rng, profile=profile)
            restart_histories.append(history)

            if best_history is None or float(history["best_loss"]) < float(best_history["best_loss"]):
                best_history = history
                best_restart = restart_index

        assert best_history is not None

        if self.config.restore_best:
            model.set_parameters(np.asarray(best_history["best_parameters"], dtype=np.float64))
            if hasattr(model, "set_classification_threshold"):
                model.set_classification_threshold(float(best_history["best_threshold"]))

        result = dict(best_history)
        result["best_restart"] = best_restart
        result["restart_histories"] = restart_histories
        return result

    def evaluate(
        self,
        model: TrainableModel,
        dataset: DatasetSplit,
        *,
        parameters: np.ndarray | None = None,
        profile: RuntimeProfile | None = None,
    ) -> dict[str, float]:
        threshold = self._current_threshold(model)
        parameter_array = model.get_parameters() if parameters is None else np.asarray(parameters, dtype=np.float64)
        backend = getattr(model, "backend", None)

        if backend is not None and hasattr(backend, "evaluate_batch"):
            with timed(profile, "train.forward_predict"):
                evaluation = backend.evaluate_batch(
                    model,
                    dataset.states,
                    dataset.labels,
                    parameter_array,
                    loss_name=self.config.loss,
                    threshold=threshold,
                )
            with timed(profile, "train.forward_loss"):
                loss = float(evaluation["loss"])
            return {"loss": loss, "accuracy": float(evaluation["accuracy"])}

        with timed(profile, "train.forward_predict"):
            probabilities = np.asarray(
                model.predict_batch(dataset.states, parameters=parameter_array),
                dtype=np.float64,
            )

        predictions = (probabilities >= threshold).astype(np.int64)
        labels_int = dataset.labels.astype(np.int64)
        accuracy = float(np.mean(predictions == labels_int))

        with timed(profile, "train.forward_loss"):
            loss = self._loss_from_probabilities(probabilities, dataset.labels)

        return {"loss": float(loss), "accuracy": accuracy}

    def gradient(
        self,
        model: TrainableModel,
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
        model: TrainableModel,
        split: DatasetSplit,
        initial_parameters: np.ndarray,
        rng: np.random.Generator,
        *,
        profile: RuntimeProfile | None = None,
    ) -> dict[str, object]:
        states = split.states
        labels = np.asarray(split.labels, dtype=np.float64)
        parameters = np.asarray(initial_parameters, dtype=np.float64).copy()

        with timed(profile, "train.initial_evaluate"):
            first_metrics = self.evaluate(model, split, parameters=parameters, profile=profile)

        current_threshold = self._current_threshold(model)

        history: dict[str, object] = {
            "loss": [first_metrics["loss"]],
            "accuracy": [first_metrics["accuracy"]],
            "threshold": [current_threshold],
            "best_loss": first_metrics["loss"],
            "best_accuracy": first_metrics["accuracy"],
            "best_parameters": parameters.copy(),
            "best_threshold": current_threshold,
            "initial_parameters": parameters.copy(),
        }

        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        optimization_step = 0

        for epoch in range(1, self.config.epochs + 1):
            for batch_indices in self._iter_minibatch_indices(labels.shape[0], rng):
                batch_states = states[batch_indices]
                batch_labels = labels[batch_indices]

                with timed(profile, "train.backward_gradient"):
                    gradient = self._loss_gradient(model, batch_states, batch_labels, parameters)

                optimization_step += 1

                with timed(profile, "train.optimizer_step"):
                    if self.config.optimizer == "adam":
                        first_moment = self.config.beta1 * first_moment + (1.0 - self.config.beta1) * gradient
                        second_moment = self.config.beta2 * second_moment + (1.0 - self.config.beta2) * (gradient**2)

                        first_unbiased = first_moment / (1.0 - self.config.beta1**optimization_step)
                        second_unbiased = second_moment / (1.0 - self.config.beta2**optimization_step)

                        parameters = parameters - self.config.learning_rate * first_unbiased / (
                            np.sqrt(second_unbiased) + self.config.epsilon
                        )
                    else:
                        parameters = parameters - self.config.learning_rate * gradient

                    model.set_parameters(parameters)

            with timed(profile, "train.threshold_update"):
                self._maybe_update_classification_threshold(model, split, parameters)

            with timed(profile, "train.epoch_evaluate"):
                metrics = self.evaluate(model, split, parameters=parameters, profile=profile)

            history["loss"].append(metrics["loss"])
            history["accuracy"].append(metrics["accuracy"])
            history["threshold"].append(self._current_threshold(model))

            if metrics["loss"] < history["best_loss"]:
                history["best_loss"] = metrics["loss"]
                history["best_accuracy"] = metrics["accuracy"]
                history["best_parameters"] = parameters.copy()
                history["best_threshold"] = self._current_threshold(model)

        history["final_parameters"] = parameters.copy()
        history["final_threshold"] = self._current_threshold(model)
        return history

    def _loss_gradient(
        self,
        model: TrainableModel,
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
                        loss_name=self.config.loss,
                    ),
                    dtype=np.float64,
                )
            except TypeError:
                if self.config.loss != "bce":
                    raise
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
        model: TrainableModel,
        states: np.ndarray,
        labels: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        gradient = np.zeros_like(parameters)

        for index in range(parameters.size):
            offset = np.zeros_like(parameters)
            offset[index] = self.config.finite_difference_eps
            loss_plus = self._objective_loss(model, states, labels, parameters + offset)
            loss_minus = self._objective_loss(model, states, labels, parameters - offset)
            gradient[index] = (loss_plus - loss_minus) / (2.0 * self.config.finite_difference_eps)

        return gradient

    def _objective_loss(
        self,
        model: TrainableModel,
        states: np.ndarray,
        labels: np.ndarray,
        parameters: np.ndarray,
    ) -> float:
        if hasattr(model, "loss"):
            try:
                return float(model.loss(states, labels, parameters=parameters, loss_name=self.config.loss))
            except TypeError:
                if self.config.loss != "bce":
                    raise
                return float(model.loss(states, labels, parameters=parameters))

        probabilities = np.asarray(model.predict_batch(states, parameters=parameters), dtype=np.float64)
        return self._loss_from_probabilities(probabilities, labels)

    def _loss_from_probabilities(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        probs = np.asarray(probabilities, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.float64)

        if self.config.loss == "mse":
            return float(np.mean((probs - labels_array) ** 2))

        clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
        return float(
            -np.mean(
                labels_array * np.log(clipped)
                + (1.0 - labels_array) * np.log(1.0 - clipped)
            )
        )

    def _iter_minibatch_indices(
        self,
        num_examples: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        if self.config.batch_size is None or self.config.batch_size >= num_examples:
            return [np.arange(num_examples, dtype=np.int64)]

        indices = np.arange(num_examples, dtype=np.int64)
        rng.shuffle(indices)
        return [
            indices[start : start + int(self.config.batch_size)]
            for start in range(0, num_examples, int(self.config.batch_size))
        ]

    def _initialize_model_threshold(self, model: object) -> None:
        if hasattr(model, "set_classification_threshold"):
            model.set_classification_threshold(self.config.classification_threshold)

    def _current_threshold(self, model: object) -> float:
        if hasattr(model, "get_classification_threshold"):
            return float(model.get_classification_threshold())
        return float(self.config.classification_threshold)

    def _maybe_update_classification_threshold(
        self,
        model: object,
        split: DatasetSplit,
        parameters: np.ndarray,
    ) -> None:
        if self.config.threshold_update == "none" or not hasattr(model, "set_classification_threshold"):
            return

        distances = np.abs(split.coupling_ratios - self.config.threshold_critical_ratio)
        left_indices = np.flatnonzero(split.coupling_ratios < self.config.threshold_critical_ratio)
        right_indices = np.flatnonzero(split.coupling_ratios > self.config.threshold_critical_ratio)

        selected: list[int] = []
        if left_indices.size > 0:
            selected.append(int(left_indices[np.argmin(distances[left_indices])]))
        if right_indices.size > 0:
            selected.append(int(right_indices[np.argmin(distances[right_indices])]))
        if len(selected) < 2:
            selected = np.argsort(distances).tolist()[: min(2, split.coupling_ratios.size)]
        if not selected:
            return

        outputs = np.asarray(
            model.predict_batch(split.states[np.asarray(selected, dtype=np.int64)], parameters=parameters),
            dtype=np.float64,
        )
        model.set_classification_threshold(float(np.mean(outputs)))
