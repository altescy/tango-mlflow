import abc
from typing import Callable, Dict, Optional, Tuple, cast

import numpy
import tango
from tango.common import Registrable

from tango_mlflow.step import MLflowStep


class Model(abc.ABC, Registrable):
    @abc.abstractmethod
    def train(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError


class Preprocessor(abc.ABC, Registrable):
    @abc.abstractmethod
    def train(self, X: numpy.ndarray) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def process(self, X: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError


@Model.register("logistic_regression")
class LogisticRegression(Model):
    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        penalty: Optional[float] = None,
    ):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights: Optional[numpy.ndarray] = None
        self.bias: Optional[float] = None
        self.penalty = penalty

    @staticmethod
    def sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        return cast(numpy.ndarray, 1.0 / (1.0 + numpy.exp(-x)))

    def train(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        # init parameters
        n_samples, n_features = X.shape
        self.weights = numpy.zeros(n_features)
        self.bias = 0.0

        # gradient descent
        for step in range(self.iterations):
            assert self.weights is not None
            assert self.bias is not None

            logit = X @ self.weights + self.bias
            prob = self.sigmoid(logit)

            # compute gradients
            dw = (1 / n_samples) * X.T @ (prob - y)
            db = (1 / n_samples) * (prob - y).sum()

            # if penalty is not None, add l2 regularization
            if self.penalty:
                dw += self.penalty * self.weights

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if callback is not None:
                loss = -1 / n_samples * (y * numpy.log(prob) + (1 - y) * numpy.log(1 - prob)).sum()
                callback({"loss": loss, "step": step})

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model not trained yet")
        linear_model = X @ self.weights + self.bias
        y_prob = self.sigmoid(linear_model)
        y_pred = (y_prob > 0.5).astype(int)
        return y_pred


@Preprocessor.register("standard_scaler")
class StandardScaler(Preprocessor):
    def __init__(self) -> None:
        self.mean: Optional[numpy.ndarray] = None
        self.std: Optional[numpy.ndarray] = None

    def train(self, X: numpy.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def process(self, X: numpy.ndarray) -> numpy.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor not trained yet")
        return cast(numpy.ndarray, (X - self.mean) / self.std)


@Preprocessor.register("pass_through")
class PassThrough(Preprocessor):
    def train(self, X: numpy.ndarray) -> None:
        pass

    def process(self, X: numpy.ndarray) -> numpy.ndarray:
        return X


@Preprocessor.register("minmax_scaler")
class MinMaxScaler(Preprocessor):
    def __init__(self) -> None:
        self.min: Optional[numpy.ndarray] = None
        self.max: Optional[numpy.ndarray] = None

    def train(self, X: numpy.ndarray) -> None:
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)

    def process(self, X: numpy.ndarray) -> numpy.ndarray:
        if self.min is None or self.max is None:
            raise RuntimeError("Preprocessor not trained yet")
        return cast(numpy.ndarray, (X - self.min) / (self.max - self.min))


@tango.Step.register("load_dataset")
class LoadDataset(tango.Step):
    def run(  # type: ignore[override]
        self,
        subset: str,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        test_mask = numpy.arange(X.shape[0]) % 5 == 0
        if subset == "train":
            self.logger.info("Loading training dataset")
            return X[~test_mask], y[~test_mask]
        elif subset == "test":
            self.logger.info("Loading test dataset")
            return X[test_mask], y[test_mask]

        raise ValueError(f"Invalid subset: {subset}")


@tango.Step.register("preprocess")
class Preprocess(tango.Step):
    def run(  # type: ignore[override]
        self,
        dataset: Tuple[numpy.ndarray, numpy.ndarray],
        preprocessor: Preprocessor,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        X, y = dataset
        X = preprocessor.process(X)
        return X, y


@tango.Step.register("train_model")
class TrainModel(MLflowStep):
    def run(  # type: ignore[override]
        self,
        dataset: Tuple[numpy.ndarray, numpy.ndarray],
        model: Model,
    ) -> Model:
        def training_callback(metrics: Dict[str, float]) -> None:
            step = int(metrics.pop("step"))
            for key, value in metrics.items():
                self.mlflow_logger.log_metric(key, value, step=step)

        X, y = dataset

        self.logger.info(f"Training model with {X.shape[0]} samples")
        model.train(X, y, callback=training_callback)

        self.logger.info("Training complete")
        return model


@tango.Step.register("train_preprocessor")
class TrainPreprocessor(MLflowStep):
    def run(  # type: ignore[override]
        self,
        dataset: Tuple[numpy.ndarray, numpy.ndarray],
        preprocessor: Preprocessor,
    ) -> Preprocessor:
        X, _ = dataset

        self.logger.info(f"Training preprocessor with {X.shape[0]} samples")
        preprocessor.train(X)

        self.logger.info("Training complete")
        return preprocessor


@tango.Step.register("evaluate")
class Evaluate(tango.Step):
    FORMAT: tango.Format = tango.JsonFormat()
    MLFLOW_SUMMARY = True

    def run(  # type: ignore[override]
        self,
        dataset: Tuple[numpy.ndarray, numpy.ndarray],
        model: Model,
    ) -> Dict[str, float]:
        X, y = dataset
        y_pred = model.predict(X)
        accuracy = (y_pred == y).mean()
        metrics = {"accuracy": accuracy}

        self.logger.info(f"Accuracy: {accuracy:.2%}")
        return metrics
