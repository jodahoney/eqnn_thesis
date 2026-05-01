"""Training support for small EQNN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from eqnn.datasets.heisenberg import DatasetBundle, DatasetSplit
from eqnn.models.base import TrainableModel
from eqnn.verification.equivariance import random_su2_rotation
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
    symmetry_regularization: bool = False
    symmetry_regularization_weight: float = 0.0
    num_symmetry_regularization_samples: int = 2
    symmetry_regularization_frequency: int = 1
    symmetry_regularization_state_samples: int | None = None
    symmetry_regularization_seed: int | None = None

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
        if self.symmetry_regularization_weight < 0.0:
            raise ValueError("symmetry_regularization_weight must be non-negative")
        if self.num_symmetry_regularization_samples < 1:
            raise ValueError("num_symmetry_regularization_samples must be at least 1")
        if self.symmetry_regularization_frequency < 1:
            raise ValueError("symmetry_regularization_frequency must be at least 1")
        if self.symmetry_regularization_state_samples is not None and self.symmetry_regularization_state_samples < 1:
            raise ValueError("symmetry_regularization_state_samples must be at least 1 when provided")


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
        epoch_callback: Callable[[int, TrainableModel], dict[str, Any] | None] | None = None,
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
            history = self._fit_once(
                model,
                split,
                initial_parameters,
                rng,
                profile=profile,
                epoch_callback=epoch_callback,
            )
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
        epoch_callback: Callable[[int, TrainableModel], dict[str, Any] | None] | None = None,
    ) -> dict[str, object]:
        states = split.states
        labels = np.asarray(split.labels, dtype=np.float64)
        parameters = np.asarray(initial_parameters, dtype=np.float64).copy()

        self._active_epoch = 0
        with timed(profile, "train.initial_evaluate"):
            first_metrics = self._training_metrics(model, split, parameters, profile=profile)

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
        if self._symmetry_regularization_configured():
            history["task_loss"] = [first_metrics["task_loss"]]
            history["symmetry_penalty"] = [first_metrics["symmetry_penalty"]]
            history["weighted_symmetry_penalty"] = [first_metrics["weighted_symmetry_penalty"]]
            history["symmetry_regularization_active"] = [first_metrics["symmetry_regularization_active"]]
            history["symmetry_regularization_note"] = self._symmetry_regularization_note()
        if epoch_callback is not None:
            history["epoch_callback"] = []

        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        optimization_step = 0

        for epoch in range(1, self.config.epochs + 1):
            self._active_epoch = epoch
            if epoch_callback is not None:
                with timed(profile, "train.epoch_callback"):
                    callback_metadata = epoch_callback(epoch, model)
                if callback_metadata is not None:
                    history["epoch_callback"].append(dict(callback_metadata))

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
                metrics = self._training_metrics(model, split, parameters, profile=profile)

            history["loss"].append(metrics["loss"])
            history["accuracy"].append(metrics["accuracy"])
            history["threshold"].append(self._current_threshold(model))
            if self._symmetry_regularization_configured():
                history["task_loss"].append(metrics["task_loss"])
                history["symmetry_penalty"].append(metrics["symmetry_penalty"])
                history["weighted_symmetry_penalty"].append(metrics["weighted_symmetry_penalty"])
                history["symmetry_regularization_active"].append(metrics["symmetry_regularization_active"])

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
        if (
            not self._symmetry_regularization_enabled()
            and self.config.gradient_backend in {"auto", "exact"}
            and hasattr(model, "loss_gradient")
        ):
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

        if self.config.gradient_backend == "exact" and not self._symmetry_regularization_enabled():
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
                task_loss = float(model.loss(states, labels, parameters=parameters, loss_name=self.config.loss))
            except TypeError:
                if self.config.loss != "bce":
                    raise
                task_loss = float(model.loss(states, labels, parameters=parameters))
        else:
            probabilities = np.asarray(model.predict_batch(states, parameters=parameters), dtype=np.float64)
            task_loss = self._loss_from_probabilities(probabilities, labels)

        if not self._symmetry_regularization_enabled() or not self._symmetry_regularization_active_for_current_epoch():
            return float(task_loss)

        penalty = self._symmetry_regularization_penalty(model, states, parameters)
        return float(task_loss + self.config.symmetry_regularization_weight * penalty)

    def _training_metrics(
        self,
        model: TrainableModel,
        split: DatasetSplit,
        parameters: np.ndarray,
        *,
        profile: RuntimeProfile | None = None,
    ) -> dict[str, float | bool]:
        task_metrics = self.evaluate(model, split, parameters=parameters, profile=profile)
        if not self._symmetry_regularization_configured():
            return {
                "loss": float(task_metrics["loss"]),
                "accuracy": float(task_metrics["accuracy"]),
            }

        penalty = self._symmetry_regularization_penalty(model, split.states, parameters)
        active = self._symmetry_regularization_enabled() and self._symmetry_regularization_active_for_current_epoch()
        weighted_penalty = float(self.config.symmetry_regularization_weight * penalty) if active else 0.0
        return {
            "loss": float(task_metrics["loss"] + weighted_penalty),
            "task_loss": float(task_metrics["loss"]),
            "accuracy": float(task_metrics["accuracy"]),
            "symmetry_penalty": float(penalty),
            "weighted_symmetry_penalty": weighted_penalty,
            "symmetry_regularization_active": bool(active),
        }

    def _symmetry_regularization_configured(self) -> bool:
        return bool(self.config.symmetry_regularization)

    def _symmetry_regularization_enabled(self) -> bool:
        return bool(self.config.symmetry_regularization) and float(self.config.symmetry_regularization_weight) > 0.0

    def _symmetry_regularization_active_for_current_epoch(self) -> bool:
        active_epoch = int(getattr(self, "_active_epoch", 1))
        if active_epoch <= 0:
            return True
        frequency = int(self.config.symmetry_regularization_frequency)
        return (active_epoch - 1) % frequency == 0

    def _symmetry_regularization_note(self) -> str:
        if not self._symmetry_regularization_configured():
            return "disabled"
        if not self._symmetry_regularization_enabled():
            return "configured_with_zero_weight"
        return "finite_difference_objective_regularizer"

    def _symmetry_regularization_penalty(
        self,
        model: TrainableModel,
        states: np.ndarray,
        parameters: np.ndarray,
    ) -> float:
        if not self._symmetry_regularization_configured():
            return 0.0

        state_array = np.asarray(states, dtype=np.complex128)
        if state_array.ndim == 1:
            state_array = state_array[np.newaxis, :]
        if state_array.ndim != 2 or state_array.shape[0] == 0:
            return 0.0

        state_array = self._symmetry_regularization_state_subset(state_array)
        num_qubits = self._infer_model_num_qubits(model, state_array.shape[1])
        base_seed = 0 if self.config.symmetry_regularization_seed is None else int(
            self.config.symmetry_regularization_seed
        )
        active_epoch = int(getattr(self, "_active_epoch", 0))

        penalties: list[float] = []
        for state_index, state in enumerate(state_array):
            baseline = self._predict_probability(model, state, parameters)
            for sample_index in range(int(self.config.num_symmetry_regularization_samples)):
                rotation = random_su2_rotation(
                    num_qubits,
                    base_seed + 100_000 * active_epoch + 1_000 * state_index + sample_index,
                )
                transformed_state = rotation @ state
                transformed_prediction = self._predict_probability(model, transformed_state, parameters)
                penalties.append(float((baseline - transformed_prediction) ** 2))

        if not penalties:
            return 0.0
        return float(np.mean(np.asarray(penalties, dtype=np.float64)))

    def _symmetry_regularization_state_subset(self, states: np.ndarray) -> np.ndarray:
        requested = self.config.symmetry_regularization_state_samples
        if requested is None or int(requested) >= states.shape[0]:
            return states

        base_seed = 0 if self.config.symmetry_regularization_seed is None else int(
            self.config.symmetry_regularization_seed
        )
        active_epoch = int(getattr(self, "_active_epoch", 0))
        rng = np.random.default_rng(base_seed + 7_919 * active_epoch)
        indices = np.sort(rng.choice(states.shape[0], size=int(requested), replace=False))
        return np.asarray(states[indices], dtype=np.complex128)

    def _infer_model_num_qubits(self, model: object, dimension: int) -> int:
        if hasattr(model, "config") and hasattr(model.config, "num_qubits"):
            return int(model.config.num_qubits)
        num_qubits = int(round(np.log2(int(dimension))))
        if 1 << num_qubits != int(dimension):
            raise ValueError("could not infer num_qubits for symmetry regularization")
        return num_qubits

    def _predict_probability(
        self,
        model: TrainableModel,
        state: np.ndarray,
        parameters: np.ndarray,
    ) -> float:
        try:
            return float(model.predict(state, parameters=parameters))
        except TypeError:
            return float(model.predict(state))

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
