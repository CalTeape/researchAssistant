import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
import cv2



import tensorflow as tf


IMAGE_SIZE = (400,400)



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def main():

    model = tf.keras.models.load_model("model.keras")
    model.layers[-1].activation = None

    dir = "allImages/"

    filenames = os.listdir(dir)

    for file in filenames:

        print(dir+file)
        img = cv2.imread(dir+file)
        img = cv2.resize(img, IMAGE_SIZE)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)




        arr = np.array(img)


        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        img_arr = np.expand_dims(arr, axis=0)

        heatmap = make_gradcam_heatmap(img_arr, model, "conv2d_3")

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]*255



        # Create an image with RGB colorized heatmap
        cv2.imwrite("heatmap.jpg", jet_heatmap)
        jet_heatmap = cv2.imread("heatmap.jpg")
        jet_heatmap = cv2.resize(jet_heatmap, IMAGE_SIZE)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
        jet_heatmap = np.array(jet_heatmap)


        #reducing blue in final image because it's overpowering everything else
        jet_heatmap = np.array([x*[0.2,0.6,1] for x in jet_heatmap])


        jet_heatmap = jet_heatmap.astype(int)


        overlayed = jet_heatmap*0.5 + gray_rgb

        #normalising so rgb values don't exceed 255
        overlayed_flat = overlayed.flatten()
        normalised_flat = (overlayed_flat/max(overlayed_flat))*255
        normalised = np.reshape(normalised_flat, overlayed.shape)



        cv2.imwrite("gradCams/" + str(file), normalised)

main()