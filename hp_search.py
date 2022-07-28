# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A training script for a DenseNet.

This example is under active development and should not be forked for other models yet.
"""
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
import keras_tuner
import numpy as np
import keras
from absl import app
from absl import flags
from keras.callbacks import BackupAndRestore
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras_tuner import BayesianOptimization
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from wandb.keras import WandbCallback
from keras.layers import Resizing
from absl import flags
import sys

import keras_cv
from keras_cv.models import DenseNet121

flags.DEFINE_boolean("wandb", False, "Whether or not to use wandb.")
flags.DEFINE_string("wandb_entity", "keras-team-testing", "WandB entity to use")
flags.DEFINE_string("wandb_project", "hp-mega-search", "WandB project ID to use")
flags.DEFINE_string(
    "experiment_name", None, "Experiment name to prepend to model names"
)
flags.DEFINE_integer("batch_size", 64, "Training batch size.")
flags.DEFINE_integer("epochs", 500, "Training epochs.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

NUM_CLASSES = 101
WEIGHTS_PATH = "weights.hdf5"


def load_caltech101(height=224, width=224, batch_size=32):
    train_ds, test_ds = tfds.load(
        "caltech101", split=["train", "test"], as_supervised=True
    )
    resizing = Resizing(height=height, width=width, crop_to_aspect_ratio=True)

    train = train_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, NUM_CLASSES)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    test = test_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, NUM_CLASSES)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    return train, test


AUGMENT_LAYERS = [
    keras_cv.layers.RandomResizedCrop(
        target_size=(224, 224),
        crop_area_factor=(0.8, 1.0),
        aspect_ratio_factor=(3 / 4, 4 / 3),
    ),
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.5),
    keras_cv.layers.CutMix(),
]


@tf.function
def augment(img, label):
    inputs = {"images": img, "labels": label}
    for layer in AUGMENT_LAYERS:
        inputs = layer(inputs)
    return inputs["images"], inputs["labels"]


path_base = "./densenet/"
weights_path = path_base + WEIGHTS_PATH
tensorboard_path = path_base + "logs/"

train, test = load_caltech101()
train = train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train = train.prefetch(tf.data.AUTOTUNE)

tensorboard = TensorBoard(log_dir=tensorboard_path)
early_stopping = EarlyStopping(patience=30)
callbacks = [tensorboard, early_stopping]

model_id = 0


def get_model_name():
    global model_id
    model_id += 1
    return f"{FLAGS.experiment_name}_{model_id}"


class MegaHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        initial_learning_rate = hp.Float(
            "initial_learning_rate",
            min_value=0.0005,
            max_value=0.05,
            step=0.0005,
        )
        end_learning_rate=hp.Float(
            "initial_learning_rate",
            min_value=0.00005,
            max_value=0.0005,
            step=0.00005,
        )
        weight_decay=hp.Float(
            "weight_decay", min_value=0.0005, max_value=0.01, step=0.0005
        )

        initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        end_learning_rate = tf.cast(end_learning_rate, tf.float32)
        weight_decay = tf.cast(weight_decay, tf.float32).numpy()

        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=PolynomialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=hp.Int(
                    "decay_steps",
                    min_value=int(train.cardinality().numpy() * FLAGS.epochs / 10),
                    max_value=int(train.cardinality().numpy() * FLAGS.epochs),
                    step=int(train.cardinality().numpy() * FLAGS.epochs / 10),
                ),
                end_learning_rate=end_learning_rate,
            ),
            weight_decay=weight_decay,
        )
        model = DenseNet121(
            include_rescaling=True,
            include_top=True,
            num_classes=NUM_CLASSES,
            input_shape=(224, 224, 3),
            name=get_model_name(),
        )
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, callbacks=(), **kwargs):
        # No custom fit HP searching yet, but we could add it here
        if FLAGS.wandb:
            wandb.init(
                entity=FLAGS.wandb_entity,
                project=FLAGS.wandb_entity,
                config=hp.values,
                name=model.name,
                reinit=True,
            )
            callbacks += [WandbCallback(save_model=False)]
        history = model.fit(
            *args,
            **kwargs,
            steps_per_epoch=1,
            validation_steps=1,
            callbacks=callbacks,
        )
        if FLAGS.wandb:
            run.finish()
        return history


with tf.distribute.MirroredStrategy().scope():
    tuner = BayesianOptimization(
        MegaHyperModel(), objective="val_accuracy", max_trials=250
    )
    tuner.search_space_summary()

    tuner.search(
        train,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        callbacks=callbacks,
        validation_data=test,
    )
