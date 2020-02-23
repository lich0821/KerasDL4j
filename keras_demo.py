#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import tensorflow as tf

import keras
from keras import layers
from keras import models
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{BASE_DIR}/java/src/main/resources/demo.h5"


def train_model():
    # Load the sample data set and split into x and y data frames
    df = pd.read_csv("data.csv")

    x = df.drop(['label'], axis=1)
    y = df['label']

    # Define the keras model
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(10,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile and fit the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])
    history = model.fit(x, y, epochs=100, batch_size=16, validation_split=.2, verbose=1)

    model.save(MODEL_PATH)


def predict(x):
    # load the model, and pass in the custom metric function
    model = load_model(MODEL_PATH)
    prediction = model.predict(x)

    return prediction


if __name__ == "__main__":
    train_model()

    x = [[1, 1, 1, 0, 0, 1, 0, 0, 0, 1]]
    x = pd.DataFrame(x)

    res = predict(x)
    print(res)
