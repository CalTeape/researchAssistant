
from tensorflow.keras.applications import * #Efficient Net included here
#from tensorflow_core.python.keras import models

from dataGen import distributeData


import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import random
import pandas as pd


import tensorflow as tf
from tensorflow.keras import models
from tensorflow import keras
#import keras
from tensorflow.keras.applications import EfficientNetB4
#from tensorflow_core.python.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

"""
GPU ACCELERATION: 
https://medium.com/analytics-vidhya/installing-tensorflow-cuda-cudnn-for-nvidia-geforce-gtx-1650-ti-onwindow-10-99ca25020d6c
https://medium.com/analytics-vidhya/step-by-step-guide-to-setup-gpu-with-tensorflow-on-windows-laptop-c84634f59857#:~:text=For%20GeForce%20GTX%201650%20%3D%3E%20Cuda,(compute%20compatibility)%20are%20compatible.


FOR EFFICIENT NET NEED 

tensorflow 2.3.0 import all keras related things from tensorflow as per above
CUDA v10.1
CUDNN 8.05


create virtual environment with python 3.7 and tensorflow version 1.15

meetings monday 11am


QUESTIONS:      https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    1.  What is activation function? How to choose      https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c#:~:text=In%20a%20binary%20classifier%2C%20we,with%20one%20node%20per%20class.
    2.  What is loss function? How to choose            https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9
    3.  What is optimizer? How to choose                https://medium.com/mlearning-ai/optimizers-in-deep-learning-7bf81fed78a0
    4.  What is dropout rate? What does it mean "to prevent overfitting"    https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9

    Ask Lech for deep
    Make bad and good same size with even numbers below 0s and above 2s in bad
    Do efficientNet B3+ and version 2

"""


IMG_SIZE = 380  #this is determined by what efficient net you use, specified at https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
BATCH_SIZE = 32
NUM_EPOCHS = 60 #picked because "Note that the convergence may take up to 50 epochs depending on choice of learning rate."

def unfreeze_model(model):

    for layer in model.layers[-50:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable=True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )


def build_model():
    seq = models.Sequential()

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    #input layer

    x = seq(inputs)

    #model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")   #include_top=False so we can use our own output (top) layers
    model = EfficientNetB4(include_top=False, input_tensor=x, weights="imagenet", drop_connect_rate=0.5)

    model.trainable=False   #freezing the core of the model so that it sticks with pretrained weights

    #building top layer
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5, name="top_dropout")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x) #activation=sigmoid because binary - https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c#:~:text=In%20a%20binary%20classifier%2C%20we,with%20one%20node%20per%20class.

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


    
def main():

    print("keras version:   " + str(keras.__version__))
    print("tensorflow version:  " + str(tf.__version__))
    random_seed=1   #must be an integer, to be used in the splitting of dataset into training and testing

    distributeData(random_seed)

    print(tf.config.experimental.list_physical_devices('GPU'))

    model = build_model()

    print(model.summary())

    unfreeze_model(model)

    print(model.summary())

    train = ImageDataGenerator(rescale=1/255)
    validation = ImageDataGenerator(rescale=1/255)
    test = ImageDataGenerator(rescale=1/255)

    train_dataset = train.flow_from_directory("data/training/",
                                                target_size= (IMG_SIZE, IMG_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode='binary')

    validation_dataset = validation.flow_from_directory("data/validation/",
                                                target_size= (IMG_SIZE, IMG_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode='binary')


    test_dataset = test.flow_from_directory("data/testing",
                                            target_size=(IMG_SIZE,IMG_SIZE),
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary')




    model_fit = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=validation_dataset, callbacks=[EarlyStopping(monitor='accuracy', patience = 10)])


    history_df = pd.DataFrame(model_fit.history)

    with open("history.csv", mode ='w') as f:
        history_df.to_csv(f)

    test_predictions = model.predict_generator(test_dataset)
    test_predicted_classes = [round(x[0]) for x in test_predictions]
    test_true = test_dataset.classes


    print(test_predictions)
    print(test_predicted_classes)
    print(test_true)


    print("Accuracy:",metrics.accuracy_score(test_true, test_predicted_classes)) #using testing answers to see how accurate those predictions were

    confusion_matrix = metrics.confusion_matrix(test_true, test_predicted_classes)   #getting confusion matrix

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=["bad", "good"])
    cm_display.plot()
    plt.show()

main()
