"""
author: Akash Singh (AIY237582)
"""

import sys
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import segmentation_models as sm
import os
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Activation, Flatten, Dense, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Conv2DTranspose, UpSampling2D, concatenate, Dropout, BatchNormalization
from keras.initializers import RandomNormal
from tensorflow.keras.models import Model



def load_data(time1_dir:str, time2_dir:str):
    # ordered images and label
    time1_images = [] 
    time2_images = []

    # Get all files in the time1 directory
    files = os.listdir(time1_dir) # list
    img_files = [file for file in files if file.endswith('.png')]  # only PNG files
    
    for img_name in tqdm(img_files):
        time1_img_pth = os.path.join(time1_dir, img_name)
        time2_img_pth = os.path.join(time2_dir, img_name)
        
        # Load image
        img_t1 = Image.open(time1_img_pth)
        img_t2 = Image.open(time2_img_pth)
        
        # convert to ndarray
        img_t1 = np.array(img_t1)
        img_t2 = np.array(img_t2)

        # scale each img to 0-1 range; normalizing each channels independently
        scaler = MinMaxScaler()
        num_channels = img_t1.shape[-1]
        img_t1 = scaler.fit_transform(img_t1.reshape(-1, num_channels)).reshape(img_t1.shape) # each column represent a band
        img_t2 = scaler.fit_transform(img_t2.reshape(-1, num_channels)).reshape(img_t2.shape) # each column represent a band
    
    
        time1_images.append(img_t1)
        time2_images.append(img_t2)
        
    return time1_images, time2_images, img_files
    
def label_prediction(data_dir, fname):
    time1_dir = data_dir + "time1/"
    time2_dir = data_dir + "time2/"
    
    test1_images, test2_images, img_files = load_data(time1_dir, time2_dir)
    print(len(test1_images), len(test2_images))
    # convert to array
    test1_images, test2_images = np.array(test1_images), np.array(test2_images)
    print(test1_images.shape, test2_images.shape)
    
    # load the best model
    model = tf.keras.models.load_model(fname, compile = False)
    
    BACKBONE = fname.split('_')[-1].split('.')[0]
    
    # preprocess input
    preprocess_input = sm.get_preprocessing(BACKBONE)
    X_test1 = preprocess_input(test1_images)
    X_test2 = preprocess_input(test2_images)
    print(X_test1.shape, X_test2.shape)
    
    y_pred = model.predict([X_test1, X_test2], batch_size=8, verbose=2).squeeze(axis=-1)
    
    # Convert predicted score to binary labels (0 or 1)
    y_pred = (y_pred > 0.5).astype(int)
    
    # Replace 0s with 0 and 1 with 255; # convert pixel label to value
    y_pred[y_pred == 0] = 0
    y_pred[y_pred == 1] = 255
    
    # To find unique values of each all pixels in all images
    unique_values_test = np.unique(np.array(y_pred).flatten())
    print(len(unique_values_test), unique_values_test)
    print(y_pred.shape)
    
    # Save as image
    
    # Create a folder to save the images if it doesn't exist
    output_folder = './assignment2Data/testLabelNotProvided/cdPredictions/'
    os.makedirs(output_folder, exist_ok = True)
    print('Predicted image labels will be saved in directory : ', output_folder)
        
    for i in range(y_pred.shape[0]):
        # Create a PIL image from the predicted labels
        img = Image.fromarray(y_pred[i].astype(np.uint8))
    
        filename = img_files[i] # name as in order while loading above
        fname = os.path.join(output_folder, filename)
        
        # Save the image as PNG
        img.save(fname)
    
    print(f'{i+1} Predicted Images label saved !!')



if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    p.add_argument('--data_dir',type=str, default="./assignment2Data/testLabelNotProvided/", help='path of testLabelNotProvided folder')
    p.add_argument('--fname',type=str, default='./model_checkpoint/cd_seresnet34.hdf5',help='path of trained model')

    args = p.parse_args()

    label_prediction(**vars(args))
