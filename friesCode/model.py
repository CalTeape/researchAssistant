from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import re
import random
import pandas as pd

from keras.layers.core import Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

from dataGen import distributeData



def buildModel():
 
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape=(500,500,3), 
            kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.MaxPool2D(2,2),
            Dropout(0.5),
            #new layer
            tf.keras.layers.Conv2D(32,(3,3), activation = 'relu',
            kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.MaxPool2D(2,2),
            Dropout(0.5),

            tf.keras.layers.Conv2D(64,(3,3), activation = 'relu',
            kernel_regularizer=regularizers.l2(0.0001)),
            tf.keras.layers.MaxPool2D(2,2),
            Dropout(0.5),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(512, activation='relu',
            kernel_regularizer=regularizers.l2(0.0001)),
            Dropout(0.5),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )

    model.compile(loss = 'binary_crossentropy',
                    optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])

    return model
   

def main():

    random_seed=1   #must be an integer, to be used in the splitting of dataset into training and testing


    #distributeData(random_seed)

    train = ImageDataGenerator(rescale=1/255)
    validation = ImageDataGenerator(rescale=1/255)
    test = ImageDataGenerator(rescale = 1/255)

    train_dataset = train.flow_from_directory("data/training/",
                                                target_size= (500,500),
                                                batch_size = 64,
                                                class_mode='binary')

    validation_dataset = validation.flow_from_directory("data/validation/",
                                                target_size= (500,500),
                                                batch_size = 64,
                                                class_mode='binary')

    test_dataset = test.flow_from_directory("data/testing",
                                            target_size=(500,500),
                                            batch_size=64,
                                            class_mode='binary')

    print(train_dataset.class_indices)


    model = buildModel()

    model_fit = model.fit(train_dataset, epochs=50, validation_data=validation_dataset, callbacks=[EarlyStopping(monitor='val_loss', patience = 3)])


    history_df = pd.DataFrame(model_fit.history)

    with open("history.csv", mode ='w') as f:
        history_df.to_csv(f)


    prediction_validation = model.predict(validation_dataset)

    predicted_classes_val = np.argmax(prediction_validation, axis=1)

    true_validation = validation_dataset.classes


    print(predicted_classes_val)
    print(true_validation)


    #print("Accuracy:",metrics.accuracy_score(true_validation, predicted_classes_val)) 

    print("F1 SCORE:",metrics.f1_score(true_validation, predicted_classes_val))

    confusion_matrix = metrics.confusion_matrix(true_validation, predicted_classes_val)   

    labels = ["bad", "good"]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=labels)
    cm_display.plot()
    plt.show()

    prediction = model.predict(test_dataset)

    predicted_classes = np.argmax(prediction, axis=1)

    true = test_dataset.classes

    print(predicted_classes)
    print(true)


    print("F1 SCORE:",metrics.f1_score(true, predicted_classes)) #using testing answers to see how accurate those predictions were

    confusion_matrix = metrics.confusion_matrix(true, predicted_classes)   #getting confusion matrix

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=labels)
    cm_display.plot()
    plt.show()

main()


                                    
