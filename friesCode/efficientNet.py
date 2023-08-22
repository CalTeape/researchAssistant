
from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib

from dataGen import distributeData


print(device_lib.list_local_devices())


"""
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


IMG_SIZE = 528  #this is determined by what efficient net you use, specified at https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
BATCH_SIZE = 64
NUM_EPOCHS = 60 #picked because "Note that the convergence may take up to 50 epochs depending on choice of learning rate."



def build_model():
    seq = models.Sequential()

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))    #input layer

    x = seq(inputs)

    #model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")   #include_top=False so we can use our own output (top) layers
    model = EfficientNetB6(include_top=False, input_tensor=x, weights="imagenet", drop_connect_rate=0.5) 

    model.trainable=False   #freezing the core of the model so that it sticks with pretrained weights

    #building top layer
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate=0.5
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)


    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x) #activation=sigmoid because binary - https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c#:~:text=In%20a%20binary%20classifier%2C%20we,with%20one%20node%20per%20class.

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


    
def main():

    random_seed=1   #must be an integer, to be used in the splitting of dataset into training and testing

    distributeData(random_seed)

    model = build_model()

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
                                            target_size=(500,500),
                                            batch_size=64,
                                            class_mode='binary')



    model_fit = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=validation_dataset)

    history_df = pd.DataFrame(model_fit.history)

    with open("history.csv", mode ='w') as f:
        history_df.to_csv(f)

    test_predictions = model.predict(test_dataset)
    test_predicted_classes = np.argmax(test_predictions, axis=1)
    test_true = test_dataset.classes

    print(test_predicted_classes)
    print(test_true)


    print("Accuracy:",metrics.accuracy_score(test_true, test_predicted_classes)) #using testing answers to see how accurate those predictions were

    confusion_matrix = metrics.confusion_matrix(test_true, test_predicted_classes)   #getting confusion matrix

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=["bad", "good"])
    cm_display.plot()
    plt.show()
main()
