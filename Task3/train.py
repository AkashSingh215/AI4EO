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

def model_train_pipeline(X_train, y_train, X_valid, y_valid, BACKBONE = 'resnet34', epochs = 100):
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # preprocess input
    X_train = preprocess_input(X_train)
    print(X_train.shape, y_train.shape)
    
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    n_classes = y_train.shape[3]

    # define model;  encoder_freeze=False i.e training Encoder with imagenet as initial weights o/w Fine tuning if True 
    base_model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes= n_classes, activation='softmax',
                   encoder_freeze=False, decoder_filters=(256, 128, 64, 32, 16), decoder_use_batchnorm=True)

    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)

    # based on this i have manually put the class weightage which sums to one; i can also take from the above one
    class_weights = [0.22,0.16,0.4] # If some class has a weight of 0.00, it means that the model will not consider that class during the calculation of the loss function.
    
    #Define loss, metrics and optimizer to be used for training
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)  
    
    metrics = ['accuracy', sm.metrics.IOUScore(class_weights=class_weights)]
    
    lr = 0.0001
    optim = keras.optimizers.Adam(lr)

    model.compile(optimizer=optim, loss= total_loss, metrics=metrics)
    
    
    model_checkpoint_folder = './model_checkpoint/'
    os.makedirs(model_checkpoint_folder, exist_ok = True)

    filename = f'train_{BACKBONE}.hdf5'
    print('Model will be saved in directory : ', model_checkpoint_folder)
    fname = os.path.join(model_checkpoint_folder, filename)

    # Set callback functions to early stop training and save the best model so far; `iou_score` is the above defined metric name
    callbacks = [ModelCheckpoint(filepath = fname, monitor = 'iou_score', mode='max', save_best_only = True, verbose = 1)]#,lrate_scheduler                         


    history = model.fit(X_train, y_train, 
                        batch_size = 16,
                        validation_data = (X_valid, y_valid),
                        verbose=1, 
                        epochs=epochs, 
                        shuffle=True,
                       callbacks = callbacks)

    return fname, history

def acc_loss_plot(history):
    # Set the size of the figure
    plt.figure(figsize=(12, 4)) 

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation IoU
    plt.subplot(1, 2, 2)
    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    plt.plot(epochs, acc, 'y', label='Training IoU')
    plt.plot(epochs, val_acc, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.show()
    
# Define U-Net model with batch normalization
def unet_model(IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CHANNELS=1, n_classes=1, kernel_initializer = 'he_normal', 
              kernel_regularizer=tf.keras.regularizers.l1(), activation='relu'):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Downsample
    c1 = Conv2D(16, (3, 3), activation=activation, kernel_initializer= kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.4)(c1) 
    c1 = Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.3)(c2) 
    c2 = Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.5)(c4)
    c4 = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c5)
    c5 = BatchNormalization()(c5)
    
    # UPSample 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.5)(c6)
    c6 = Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c6)
    c6 = BatchNormalization()(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c7)
    c7 = BatchNormalization()(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.3)(c8)
    c8 = Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c8)
    c8 = BatchNormalization()(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.4)(c9)
    c9 = Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same', kernel_regularizer =
                kernel_regularizer)(c9)
    c9 = BatchNormalization()(c9)
     
    logits = Conv2D(n_classes, (1, 1))(c9) # No activation function
    outputs = Softmax()(logits) # Add a separate layer for softmax
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def train_scratch(X_train, y_train, X_valid, y_valid, model, model_name):
    #Define loss, metrics and optimizer to be used for training
    class_weights = [0.22,0.16,0.4] # If some class has a weight of 0.00, it means that the model will not consider that class during the calculation of the loss function.
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)  

    metrics = ['accuracy', sm.metrics.IOUScore(class_weights=class_weights)]

    lr = 0.001
    optim = keras.optimizers.Adam(lr)

    model.compile(optimizer=optim, loss= total_loss, metrics=metrics)
    
    model_checkpoint_folder = './model_checkpoint/'
    os.makedirs(model_checkpoint_folder, exist_ok = True)

    filename = f'train_scratch_{model_name}.hdf5'
    print('Model will be saved in directory : ', model_checkpoint_folder)
    fname_scratch = os.path.join(model_checkpoint_folder, filename)

    # Set callback functions to early stop training and save the best model so far
    callbacks = [ModelCheckpoint(filepath = fname_scratch, monitor = 'iou_score', mode='max', save_best_only = True, verbose = 1)]#,lrate_scheduler                         


    history_scratch = model.fit(X_train, y_train, 
                        batch_size = 16,
                        validation_data = (X_valid, y_valid),
                        verbose=1, 
                        epochs=200, 
                        shuffle=True,
                       callbacks = callbacks)
    
    return fname_scratch, history_scratch


    
def train(data_dir):
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

    # Pre-trained model
    BACKBONE = 'seresnet34'
    epochs = 200
    fname_ser34, history_ser34 = model_train_pipeline(X_train, y_train, X_valid, y_valid, BACKBONE = BACKBONE, epochs = epochs)
    acc_loss_plot(history_ser34)

    # from scratch
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    n_classes = y_train.shape[3]

    print(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, n_classes)

    model_name = 'model1'
    model1 = unet_model(IMG_HEIGHT=IMG_HEIGHT , IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, n_classes=n_classes, kernel_initializer =
                        'he_normal', kernel_regularizer=tf.keras.regularizers.l1(), activation='relu')

    fname_model1, history_model1 = train_scratch(X_train, y_train, X_valid, y_valid, model1, model_name)

    acc_loss_plot(history_model1)

    model_name = 'model2'
    model2 = unet_model(IMG_HEIGHT=IMG_HEIGHT , IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, n_classes=n_classes, kernel_initializer =
                        'he_normal', kernel_regularizer=tf.keras.regularizers.l1_l2(), activation='relu')

    fname_model2, history_model2 = train_scratch(X_train, y_train, X_valid, y_valid, model2, model_name)

    acc_loss_plot(history_model2)

    model_name = 'model3'
    model3 = unet_model(IMG_HEIGHT=IMG_HEIGHT , IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, n_classes=n_classes, kernel_initializer =
                        'glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(), activation='tanh')

    fname_model3, history_model3 = train_scratch(X_train, y_train, X_valid, y_valid, model3, model_name)

    acc_loss_plot(history_model3)

    model_name = 'model4'
    model4 = unet_model(IMG_HEIGHT=IMG_HEIGHT , IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, n_classes=n_classes, kernel_initializer =
                        'orthogonal', kernel_regularizer=tf.keras.regularizers.l1_l2(), activation=tf.keras.layers.LeakyReLU())

    fname_model4, history_model4 = train_scratch(X_train, y_train, X_valid, y_valid, model4, model_name)

    acc_loss_plot(history_model4)
    
    print('Process completed !!')




if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    p.add_argument('--data_dir',type=str, default="./assignment3DataUpload/", help='path of dataset folder')

    args = p.parse_args()

    train(**vars(args))
