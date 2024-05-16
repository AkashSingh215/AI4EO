"""
author: Akash Singh (AIY237582)
"""

import sys
import argparse
import gc
from PIL import Image
import tensorflow as tf
import keras
from keras import backend as K
import osgeo
import segmentation_models as sm
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, Softmax
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.metrics import MeanIoU



def load_data(images_dir:str, groundtruth_dir:None):
    images = []
    masks = []

    # Get all image files in the directory
    files = os.listdir(images_dir) # list
    img_files = [file for file in files if file.endswith('.tif')]  # only tif files
    
    for img_name in tqdm(img_files):
        img_pth = os.path.join(images_dir, img_name)
        
        # Load image
        image_data = gdal.Open(img_pth)
        img = image_data.ReadAsArray()  # shape (channels, X, Y); has 1 channel only
        
        # scale img to 0-1 range; normalizing each channels independently
        scaler = MinMaxScaler()
        num_channels = 1
        img = scaler.fit_transform(img.reshape(-1, num_channels)).reshape(img.shape) # each column represent a band
        
        images.append(img)
        
        
        if groundtruth_dir:
            mask_path = os.path.join(groundtruth_dir, img_name)
        
            # Load mask
            mask_data = gdal.Open(mask_path)
            mask = mask_data.ReadAsArray()  # shape (channels, X, Y)
            mask = mask.transpose(1,2,0) # shape (X, Y, channels)
            
            # Set values less than 200 to 0 and greater than 200 to 255; as one label has many values b/w 0-255
            mask[mask <= 200] = 0
            mask[mask > 200] = 255

            masks.append(mask)
        
    print(f"Images count: {len(images)} & Masks count: {len(masks)}")
    print('Images are Normalized and Masks are not !')

    print()
    # to get the shape of all images and masks
    if groundtruth_dir:
        print('Each Image & its corresponding mask size : ')
        for i,m in zip(images, masks):
            print(i.shape, m.shape)

    else:
        print('Each Image size : ')
        for i in images:
            print(i.shape)
            
    print()
        
    return images, masks, img_files

# creating overlap patches and which makes train data larger
def extract_patches(images:list, masks:list, patch_size=(256, 256), stride=1): 
    image_patches = []
    mask_patches = []

    for i in tqdm(range(len(images))):
        img = images[i]
        mask = masks[i]
        height, width = img.shape

        for y in range(0, height - patch_size[0] + 1, stride):
            for x in range(0, width - patch_size[1] + 1, stride):
                image_patch = img[y:y + patch_size[0], x:x + patch_size[1]]
                mask_patch = mask[y:y + patch_size[0], x:x + patch_size[1], :]

                image_patches.append(image_patch)
                mask_patches.append(mask_patch)
    
    return np.array(image_patches), np.array(mask_patches)

def rgb_to_label(image_mask):
    """
    Convert 3D- RGB label masks to 2D label masks.
    """
    # below is the RGB to integer label
    C0 = np.array([0,0,0]) # unseen; black
    C1 = np.array([255, 0, 0])  # red
    C2 = np.array([0, 255, 0]) # green
    C3 = np.array([255, 255, 0]) # yellow
    
    # integer label
    label_seg = np.zeros(image_mask.shape[:2], dtype=np.uint8)
    label_seg[np.all(image_mask == C1, axis=-1)] = 0
    label_seg[np.all(image_mask == C2, axis=-1)] = 1
    label_seg[np.all(image_mask == C3, axis=-1)] = 2
    label_seg[np.all(image_mask == C0, axis=-1)] = 3
    
    return label_seg

def temperature_scaling(y_logits, y_valid):
    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32, name="temp")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    for epoch in range(500):
        with tf.GradientTape() as tape:
            scaled_logits = y_logits / temp
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_valid, logits=scaled_logits))
            gradients = tape.gradient(loss, [temp])
        
        optimizer.apply_gradients(zip(gradients, [temp]))
        
        if epoch % 100 == 0:
            print("Epoch:", (epoch + 1), "Loss =", "{:.15f}".format(loss.numpy()), "Temperature =", temp.numpy())
    
    return temp.numpy()

def best_temp(models: dict, X_valid, y_valid):
    temps = []
    
    for name, base_model in models.items(): # base_model is with softmax
        if name == 'seresnet34':
            # preprocess input
            preprocess_input = sm.get_preprocessing('seresnet34') # if in the models list we also use pre-trained model here seresnet
            X_valid = preprocess_input(X_valid)


        logits_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        y_logits = logits_model.predict(X_valid, verbose=0)  # logits: shape (bs, height, width, num_classes)
        
        final_temp_value = temperature_scaling(y_logits, y_valid)
        temps.append(final_temp_value)
        
    return temps


def assign_labels(softmax_scores_list, threshold=0.005):
    assigned_labels_list = []
    
    # Step 1: Assign labels based on maximum score difference from 2nd max score for each model
    for softmax_scores in softmax_scores_list:
        max_indices = np.argmax(softmax_scores, axis=-1)
        max_values = np.max(softmax_scores, axis=-1)
        
        sorted_scores = np.sort(softmax_scores, axis=-1)
        second_max_values = sorted_scores[..., -2]
        
        labels = np.where(max_values - second_max_values > threshold, max_indices.squeeze(), -1) # -1 is label for 'OOD'
        assigned_labels_list.append(labels)
    
    # Step 2: Find class with maximum repetition for each pixel
    assigned_labels_list = np.stack(assigned_labels_list, axis=-1)
    assigned_labels_list = np.squeeze(assigned_labels_list, axis=0)  # Remove the batch dimension
    
    # Step 2: Find class with maximum repetition for each pixel
    final_labels = np.zeros((assigned_labels_list.shape[0], assigned_labels_list.shape[1], 1), dtype=np.int8) # np.int8 as -1 label is there so no np.uint8
    
    # Iterate through each pixel
    for row in range(assigned_labels_list.shape[0]):
        for col in range(assigned_labels_list.shape[1]):
            pixel_labels = assigned_labels_list[row, col]
            unique_labels, label_counts = np.unique(pixel_labels, return_counts=True)
            
            # If all labels are different, assign 'OOD'
            if len(np.unique(label_counts)) == 1:
                final_labels[row, col, 0] = -1 # -1 is label for 'OOD'
                
            else:
                # Majority voting
                majority_label = unique_labels[np.argmax(label_counts)]
                final_labels[row, col, 0] = majority_label
    
    return np.array(final_labels)

def test_prediction(models:dict, images:list, temps : list, patch_size=(256, 256)): 
    # Order of models and temps should be same
    pred_labels = []

    # if in the models list we also use pre-trained model here seresnet
    preprocess_input = sm.get_preprocessing('seresnet34')
    
    for i in tqdm(range(len(images))):
        img = images[i] # each image is of size (height, width, channels)
        label_seg = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8)
        height, width, _ = img.shape
        
        if height % patch_size[0] == 0: # say height = 448 and patch_size = (224,224)
            height_pixels = [i for i in range(0, height - patch_size[0] + 1, patch_size[0])] # then = [0, 224]

        else: # say height = 449
            height_pixels = [i for i in range(0, height + 1, patch_size[0])] # then = [0, 224, 448]


        if width % patch_size[1] == 0: # slly
            width_pixels = [i for i in range(0, width - patch_size[1] + 1, patch_size[1])]

        else: # say height = 449
            width_pixels = [i for i in range(0, width + 1, patch_size[1])]
            

        for y in height_pixels:
            for x in width_pixels:
                x1 = x
                y1 = y

                y2 = y1 + patch_size[0]
                x2 = x1 + patch_size[1]

                if width % patch_size[1] != 0 and x == width_pixels[-1]: # if not multiple of patch size & x is the last element
                    # then take patch from last-patch_size to last pixel
                    x1 = width - patch_size[1]
                    x2 = width

                    width_rem = width - x # difference between last pixel in image and last element in width_pixels

                if height % patch_size[0] != 0 and y == height_pixels[-1]: # if not multiple of patch size& y is the last element
                    # then take patch from last-patch_size to last pixel
                    y1 = height - patch_size[0]
                    y2 = height

                    height_rem = height - y # slly


                image_patch = img[y1:y2, x1:x2, :]

                image_patch = np.expand_dims(image_patch, 0) # to shape (1, height, width, channels)

                softmax_scores_list = []
                for index, (name, logits_model) in enumerate(models.items()): # base_model is with softmax
                    if name == 'seresnet34':
                        # preprocess input
                        image_patch = preprocess_input(image_patch)


                    # logits_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
                    y_logits = logits_model.predict(image_patch, verbose=0)  # logits: shape (1, height, width, num_classes)
                    y_logits_temp = np.divide(y_logits, temps[index]) # logits after dividing with final temperature, temp
                    # Apply softmax activation
                    y_pred = tf.nn.softmax(tf.convert_to_tensor(y_logits_temp)).numpy() # shape (1, height, width, num_classes)

                    softmax_scores_list.append(y_pred)

                    del logits_model
                    gc.collect()
                    K.clear_session()
                    
                final_labels = assign_labels(softmax_scores_list) # shape (height, width, 1)
                final_labels = np.squeeze(final_labels, axis=-1) # shape (height, width)
                # this will have the class integer name

                x1p = 0
                y1p = 0
                if width % patch_size[1] != 0 and x == width_pixels[-1]:
                    x1p = -width_rem

                if height % patch_size[0] != 0 and y == height_pixels[-1]:
                    y1p = - height_rem


                label_seg[y:y + patch_size[0], x:x + patch_size[1]] = final_labels[y1p:,x1p:]

        pred_labels.append(label_seg)

    return pred_labels

# for test set prediction
def test_pipeline(models, temps, test_images):
    # test set prediction: this will have class label number
    patch_size = (256, 256)

    # Input images has to be a list with each image having shape : (height, width, channels)
    test_pred_labels = test_prediction(models, test_images, temps, patch_size)
    print("Test set prediction labels length :", len(test_pred_labels))
    print('Shape of one mask label : ', test_pred_labels[0].shape)

    return test_pred_labels


def label_to_rgb(label_mask):
    """
    Convert 2D label masks back to RGB masks.
    """
    # RGB values corresponding to each class label
    colors = {
        0: [255, 0, 0],      # Class 0
        1: [0, 255, 0],    # Class 1
        2: [255, 255, 0],   # Class 2
        -1: [0 , 0, 255]    # OOD classes
    }

    num_channels = len(colors[0])
    # an empty RGB image mask
    rgb_mask = np.zeros((label_mask.shape[0], label_mask.shape[1], num_channels), dtype=np.uint8)

    # Map class labels to RGB values
    for label, color in colors.items():
        rgb_mask[label_mask == label] = color

    return rgb_mask

# for visualizing how the test set segmentation happened
def test_segmented_mask_plot(test_pred_labels, test_images):
    test_pred_mask = []
    for label_mask in test_pred_labels: 
        pred_rgb = label_to_rgb(label_mask)
        test_pred_mask.append(pred_rgb)

    
    # Plot the predicted mask
    i = random.randint(0, len(test_pred_mask)- 1) # -1 bec. end is also included
    print(test_pred_mask[i].shape)
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth Label')
    plt.imshow(test_images[i], cmap='gray')
    # Plot for ground truth
    plt.subplot(1, 2, 2)
    plt.title('Predicted Label')
    plt.imshow(test_pred_mask[i]) # RGB
    plt.show()

    return test_pred_mask

# save the test mask
def save_mask(data_dir, pred_mask, img_files):

    # Save folder for masks
    # Create a folder to save the images if it doesn't exist
    output_folder = data_dir + 'testImagesOutput/'
    os.makedirs(output_folder, exist_ok = True)
    print('Predicted image labels will be saved in directory : ', output_folder)
    
    for i, img_name in tqdm(enumerate(img_files)):
        # Save as image .png
        label_img = pred_mask[i]
        # Create a PIL image from the predicted labels
        img = Image.fromarray(label_img.astype(np.uint8))
    
        fname = os.path.join(output_folder, img_name)
        
        # Save the image as PNG
        img.save(fname)

    print('Successfully Saved the mask !!')
    
    

def test(data_dir, model_ser34_fname, model_1_fname, model_2_fname, model_3_fname, model_4_fname):
    # train dataset folder
    train_images_dir = data_dir + 'trainingImagesAndLabels/' + "Images/"
    train_groundtruth_dir = data_dir + 'trainingImagesAndLabels/' + "Labels/"

    # train set
    train_images, train_masks,_ = load_data(train_images_dir, train_groundtruth_dir)

    idx = random.randint(0, len(train_images))
    print(f'Shape :', train_images[idx].shape, train_masks[idx].shape)

    # Plot for image
    plt.subplot(1, 3, 1)
    plt.imshow(train_images[idx], cmap='gray')
    # Plot for label
    plt.subplot(1, 3, 3)
    plt.imshow(train_masks[idx])
    plt.show()
    
    # patches extraction
    # train set:
    patch_size = (256, 256)
    stride = 80
    train_image_patches, train_mask_patches = extract_patches(train_images, train_masks, patch_size, stride)
    print("Image patches shape:", train_image_patches.shape)
    print("Mask patches shape:", train_mask_patches.shape) 
    
    # To find unique values of each all pixels in all images
    unique_values = np.unique(train_mask_patches.reshape(-1, 3), axis=0) # Reshape to 2D array;  find the unique rows (unique pixel values) - a pixel with 3 channels
    print(len(unique_values))
    print(unique_values)
    
    # Train labels
    train_labels = []
    for i in tqdm(range(train_mask_patches.shape[0])):
        label = rgb_to_label(train_mask_patches[i])
        train_labels.append(label)
    
    train_labels = np.array(train_labels)
    print(train_labels.shape)
    print('Total unique classes : ', len(np.unique(train_labels)))
    print('Class Labels : ', np.unique(train_labels))
    
    train_labels = np.expand_dims(train_labels, axis=3)
    print(train_labels.shape)

    n_classes = len(np.unique(train_labels))
    train_labels_cat = to_categorical(train_labels, num_classes=n_classes) # this gives one-hot in order refer to documentation
    # Note: The input labels has to be in number form and value from 0 to num_class-1.
    print(train_labels_cat.shape)

    # slice the array to have 3 dimension for each one-hot vector. As there are 3 actual class
    train_labels_cat = train_labels_cat[:, :, :, :3]

    # find the pixels with [0,0,0] values i.e unknown class
    zero_pixels = np.all(train_labels_cat == [0, 0, 0], axis=-1)

    # Replace label for unknown class with [0.33,0.33,0.33] i.e it belongs to all class
    train_labels_cat[zero_pixels] = [0.33, 0.33, 0.33] # assign equal probabilities to each class
    print(train_labels_cat.shape)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(np.expand_dims(train_image_patches, axis=-1), train_labels_cat, test_size=0.2,
                                                          random_state=42)

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)

    del train_images, train_masks, train_image_patches, train_mask_patches, train_labels_cat, train_labels
    gc.collect()
    K.clear_session()
    
    ## Load each models as logits generating
    model_ser34 = tf.keras.models.load_model(model_ser34_fname, compile = False)

    model_1 = tf.keras.models.load_model(model_1_fname, compile = False)
    model_2 = tf.keras.models.load_model(model_2_fname, compile = False)
    model_3 = tf.keras.models.load_model(model_3_fname, compile = False)
    model_4 = tf.keras.models.load_model(model_4_fname, compile = False)

    models = {'seresnet34': model_ser34, 'model1': model_1, 'model2': model_2, 'model3': model_3, 'model4': model_4}
    del model_ser34, model_1 , model_2, model_3, model_4
    gc.collect()
    K.clear_session()

    # final T
    temps = best_temp(models, X_valid, y_valid)

    # Testing
    ## Load each models as logits generating
    model_ser34 = tf.keras.models.load_model(model_ser34_fname, compile = False)
    model_ser34 = Model(inputs=model_ser34.input, outputs=model_ser34.layers[-2].output)

    model_1 = tf.keras.models.load_model(model_1_fname, compile = False)
    model_1 = Model(inputs=model_1.input, outputs=model_1.layers[-2].output)
    model_2 = tf.keras.models.load_model(model_2_fname, compile = False)
    model_2 = Model(inputs=model_2.input, outputs=model_2.layers[-2].output)
    model_3 = tf.keras.models.load_model(model_3_fname, compile = False)
    model_3 = Model(inputs=model_3.input, outputs=model_3.layers[-2].output)
    model_4 = tf.keras.models.load_model(model_4_fname, compile = False)
    model_4 = Model(inputs=model_4.input, outputs=model_4.layers[-2].output)

    models = {'seresnet34': model_ser34, 'model1': model_1, 'model2': model_2, 'model3': model_3, 'model4': model_4}
    del model_ser34, model_1 , model_2, model_3, model_4
    gc.collect()
    K.clear_session()

    # test dataset folder
    test_images_dir = data_dir + 'testImages/'

    test_images, _, test_img_files = load_data(test_images_dir, None)
    test_images = np.expand_dims(test_images, axis=-1)

    idx = random.randint(0, len(test_images)-1)
    print(f'Shape :', test_images[idx].shape)

    # Plot for image
    plt.imshow(test_images[idx], cmap='gray')
    plt.show()

    test_pred_labels = test_pipeline(models, temps, test_images)
    test_pred_labels[0]

    test_pred_mask = test_segmented_mask_plot(test_pred_labels, test_images)

    save_mask(data_dir, test_pred_mask, test_img_files)
    
    print('Process completed !!')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    p.add_argument('--data_dir',type=str, default="./assignment3DataUpload/", help='path of dataset folder')
    p.add_argument('--model_ser34_fname',type=str, default='./model_checkpoint/train_seresnet34.hdf5',help='path of trained models')
    p.add_argument('--model_1_fname',type=str, default='./model_checkpoint/train_scratch_model1.hdf5',help='path of trained models')
    p.add_argument('--model_2_fname',type=str, default='./model_checkpoint/train_scratch_model2.hdf5',help='path of trained models')
    p.add_argument('--model_3_fname',type=str, default='./model_checkpoint/train_scratch_model3.hdf5',help='path of trained models')
    p.add_argument('--model_4_fname',type=str, default='./model_checkpoint/train_scratch_model4.hdf5',help='path of trained models')
          
    args = p.parse_args()

    test(**vars(args))
