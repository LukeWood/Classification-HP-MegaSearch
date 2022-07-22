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

import keras_cv
from keras_cv.models import DenseNet121

NUM_CLASSES = 101
EPOCHS = 500
BATCH_SIZE = 32
WEIGHTS_PATH = "weights.hdf5"

def load_caltech101(height=200, width=200, batch_size=32):
    train_ds, test_ds = tfds.load(
        "caltech101", split=["train", "test"], as_supervised=True
    )
    resizing = Resizing(width, height)
    train = train_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, 101)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    test = test_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, 101)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    return train, test

AUGMENT_LAYERS = [
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.3),
    keras_cv.layers.RandomCutout(height_factor=0.1, width_factor=0.1),
]


@tf.function
def augment(img, label):
    inputs = {"images": img, "labels": label}
    for layer in AUGMENT_LAYERS:
        inputs = layer(inputs)
    return inputs["images"], inputs["labels"]


path_base = "./densenet/"
backup_path = path_base + "backup/"
weights_path = path_base + WEIGHTS_PATH
tensorboard_path = path_base + "logs/"

train, test = load_caltech101()
train = train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train = train.prefetch(tf.data.AUTOTUNE)

backup = BackupAndRestore(backup_path)
checkpoint = ModelCheckpoint(
    WEIGHTS_PATH,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="max",
)
tensorboard = TensorBoard(log_dir=tensorboard_path)
early_stopping = EarlyStopping(patience=10)
callbacks = [backup, checkpoint, tensorboard, early_stopping]

# Optional - for WandB integration
#wandb.init(project='mega-grid-search', entity="keras-team-testing")
#callbacks.append(WandbCallback())

class MegaHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential([
            keras_cv.layers.RandAugment(value_range=[0,1], augmentations_per_image=hp.Int("augmentations_per_image", min_value=2, max_value=5, step=1), magnitude=hp.Float("magnitude", min_value=.05, max_value=.3, step=.05)),
            DenseNet121(
                include_rescaling=True,
                include_top=True,
                num_classes=NUM_CLASSES,
                input_shape=(200, 200, 3),
            )
        ])
        model.compile(
            optimizer=Adam(
                learning_rate=PolynomialDecay(
                    initial_learning_rate=hp.Float("initial_learning_rate", min_value=.0005, max_value=.05, step=.0005),
                    decay_steps=hp.Int("decay_steps", min_value=train.cardinality().numpy() * EPOCHS/10, max_value=train.cardinality().numpy() * EPOCHS, step=train.cardinality().numpy() * EPOCHS/1),
                    end_learning_rate=hp.Float("initial_learning_rate", min_value=.00005, max_value=.0005, step=.00005),
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        # No custom fit HP searching yet, but we could add it here
        return model.fit(
            *args,
            **kwargs,
        )

with tf.distribute.MirroredStrategy().scope():
    tuner = BayesianOptimization(MegaHyperModel(), objective='val_accuracy', max_trials=250)
    tuner.search_space_summary()

    tuner.search(
        train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=test,
    )
