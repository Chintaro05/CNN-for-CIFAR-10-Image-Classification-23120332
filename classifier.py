from __future__ import annotations

import abc
import os
from typing import Optional

import numpy as np
import tensorflow as tf


class MNISTClassifier(abc.ABC):
    """Base class for MNIST digit classifiers."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return a compiled Keras model."""

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 128,
        validation_split: float = 0.1,
        callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
        augment: bool = False,
    ) -> tf.keras.callbacks.History:
        """Train the model on the given data."""
        if self.model is None:
            self.model = self.build_model()

        callbacks = callbacks or []

        if augment:
            if x_train.ndim != 4:
                raise ValueError("Data augmentation requires 4D image input data.")

            if validation_split > 0.0:
                split = int(len(x_train) * (1.0 - validation_split))
                x_train_main = x_train[:split]
                y_train_main = y_train[:split]
                x_val = x_train[split:]
                y_val = y_train[split:]
            else:
                x_train_main = x_train
                y_train_main = y_train
                x_val = None
                y_val = None

            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=15,
                fill_mode="reflect",
            )
            datagen.fit(x_train_main)

            train_generator = datagen.flow(x_train_main, y_train_main, batch_size=batch_size)
            validation_data = (x_val, y_val) if x_val is not None else None
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=max(1, len(x_train_main) // batch_size),
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=2,
            )
        else:
            history = self.model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=2,
            )

        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data."""
        if self.model is None:
            raise RuntimeError("Model is not built or loaded.")

        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred_proba = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=-1)

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "y_pred": y_pred,
        }

    def save(self, path: str) -> None:
        """Save the model to the given file path."""
        if self.model is None:
            raise RuntimeError("Model is not built or loaded.")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a model from the given file path."""
        self.model = tf.keras.models.load_model(path)


class LogisticRegressionClassifier(MNISTClassifier):
    """Logistic regression (single dense layer with softmax)."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class NeuralNetworkClassifier(MNISTClassifier):
    """Simple feedforward neural network."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CIFARMLPClassifier(MNISTClassifier):
    """Baseline MLP for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(32, 32, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CIFARCNNClassifier(MNISTClassifier):
    """Convolutional neural network for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),

            # Block 1
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.2),

            # Block 2
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.3),

            # Block 3 (added for better feature extraction)
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.4),

            # Head
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CIFARCNNAugClassifier(CIFARCNNClassifier):
    """CNN with in-model data augmentation for CIFAR-10."""

    def build_model(self) -> tf.keras.Model:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.1),
        ], name="data_augmentation")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            augmentation,

            # Block 1
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.2),

            # Block 2
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.3),

            # Block 3
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.4),

            # Head
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


class CIFARCNNImprovedClassifier(CIFARCNNAugClassifier):
    """Augmented CNN with callbacks for CIFAR-10."""

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 60,
        batch_size: int = 128,
        validation_split: float = 0.1,
    ) -> tf.keras.callbacks.History:
        if self.model is None:
            self.model = self.build_model()

        os.makedirs("models", exist_ok=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-5,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "models/cnn_improved_best.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return history
