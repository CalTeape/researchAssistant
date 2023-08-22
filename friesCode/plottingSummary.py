import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    target = "modelSummaries/model100Epochs.csv"

    df = pd.read_csv(target)

    training_loss = df["loss"]
    training_accuracy = df["accuracy"]

    validation_loss = df["val_loss"]
    validation_accuracy = df["val_accuracy"]

    #ax1 = plt.plot(validation_accuracy)
    ax1 = plt.subplot(2,2,1)
    ax1.plot(validation_accuracy)
    ax2 = plt.subplot(2,2,2)
    ax2.plot(validation_loss)

    ax3 = plt.subplot(2,2,3)
    ax3.plot(training_accuracy)
    ax4 = plt.subplot(2,2,4)
    ax4.plot(training_loss)

    ax1.set_title("validation_accuracy")
    ax2.set_title("validation_loss")

    ax2.set_ybound(0,2.5)

    ax3.set_title("training_accuracy")
    ax4.set_title("training_loss")

    
    fig = ax1.get_figure()

    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.show()

main()




