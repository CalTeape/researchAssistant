
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

from dataGen import distributeData

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
#from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

"""
GPU ACCELERATION: 
https://medium.com/analytics-vidhya/installing-tensorflow-cuda-cudnn-for-nvidia-geforce-gtx-1650-ti-onwindow-10-99ca25020d6c
https://medium.com/analytics-vidhya/step-by-step-guide-to-setup-gpu-with-tensorflow-on-windows-laptop-c84634f59857#:~:text=For%20GeForce%20GTX%201650%20%3D%3E%20Cuda,(compute%20compatibility)%20are%20compatible.

because this laptop has NVIDIA GeForce GTX 1650 ti, we need:
CUDA v10 (legacy)
CUDNN 7.4.1 (for CUDA 10)


create virtual environment with python 3.7 and tensorflow version 1.15

THINGS TO TRY:

experiment with larger pooling kernel, and multiple densely connected layers at the top of model.

do we need to try and preserve as much of the image for as many of training parameters as possible?

should the training parameters be concentrated at the end (densely connected layer before outputs), or should they be 
concentrated at the start, when image is still as big as possible?

"""


BATCH_SIZE = 32
IMAGE_SIZE = (400,400)
NUM_EPOCHS = 50

def buildModel_mobile():
    input_shape = (*IMAGE_SIZE, 3)

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True

    input_layer = tf.keras.layers.Input(shape=input_shape),
    x = base_model(input_layer, training=True)
        # x = Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.Dense(units=32, activation="relu")(x)
    x = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x)

    model.compile(
        optimizer="adam",
        loss = "binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
def buildModel_custom():
    input_shape = (*IMAGE_SIZE, 3)
    model = tf.keras.models.Sequential(
        [
            #input layer
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape,
                                   kernel_regularizer=regularizers.l2(0.0001)),

            #first layer
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Dropout(0.5),


            # 2nd layer
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Dropout(0.5),

            # 3rd layer
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                   kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Flatten(),


            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.5),

            #output layer (activation sigmoid because binary data)
            tf.keras.layers.Dense(3, activation='softmax')
        ]
    )

    # CHANGED FROM BINARY TO CATEGORICAL FOR MNIST
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    return model



def predictAllImages():
    print(tf.config.experimental.list_physical_devices('GPU'))

    model = tf.keras.models.load_model("model.keras")

    files = os.listdir("allImages/")
    predictions = []

    for file in files:
        path = "allImages/" + file
        img = cv2.imread(path)
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)  #expanding so that we have batches of 1 (model will only taken 4 dim vector)

        probs = model.predict(img)
        class_prediction = np.argmax(probs)

        if class_prediction == 0:
            predictions.append(">2")
        elif class_prediction == 1:
            predictions.append("1-2")
        else:
            predictions.append("<1")


    df = pd.DataFrame(zip(files, predictions), columns=["filename", "prediction"])
    print(df)

    df.to_csv("predictions.csv")
def main():

    random_seed=1   #must be an integer, to be used in the splitting of dataset into training and testing


    #distributeData(random_seed)

    print(tf.config.experimental.list_physical_devices('GPU'))

    train = ImageDataGenerator(rescale=1/255)
    validation = ImageDataGenerator(rescale=1/255)
    test = ImageDataGenerator(rescale = 1/255)

    train_dataset = train.flow_from_directory("data/training/",
                                                target_size= IMAGE_SIZE,
                                                batch_size = BATCH_SIZE,
                                                class_mode='categorical')

    validation_dataset = validation.flow_from_directory("data/validation/",
                                                target_size= IMAGE_SIZE,
                                                batch_size = BATCH_SIZE,
                                                class_mode='categorical')

    test_dataset = test.flow_from_directory("data/testing",
                                            target_size=IMAGE_SIZE,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            class_mode='categorical')


    print(train_dataset.class_indices)

    filenames = test_dataset.filenames
    true = test_dataset.classes


    if True:
        model = buildModel_custom()
        print(model.summary())

        model_fit = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=validation_dataset, callbacks=[EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights=True)])
        model.save("model.keras")

        history_df = pd.DataFrame(model_fit.history)
        with open("history.csv", mode='w') as f:
            history_df.to_csv(f)

    else:
        model = tf.keras.models.load_model("model.keras")


    eval_validation = model.evaluate(validation_dataset, verbose=1)
    print(eval_validation)

    eval_test = model.evaluate(test_dataset, verbose=1)

    print(eval_test)

    pred_test = model.predict(test_dataset)

    pred_classes_test = np.argmax(pred_test, axis=1)


    #df = pd.DataFrame(zip(filenames, true, pred_classes_test), columns=["filename", "true", "prediction"])
    #df.to_csv("predictions.csv")


    #for i in range(len(pred_classes_test)):
    #    if pred_classes_test[i] != true[i]:
    #        print("incorrectly labelled: " + filenames[i] + ", labelled as " + str(pred_classes_test[i]))


    print("Accuracy:",metrics.accuracy_score(true, pred_classes_test))

    confusion_matrix = metrics.confusion_matrix(true, pred_classes_test)

    labels = ["bad", "good", "zeroString"]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=labels)
    cm_display.plot()
    plt.show()



predictAllImages()


                                    
