import os
import numpy as np
import h5py
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def mask_clouds(h5_file, lower_threshold=205, upper_threshold=220):
    """Create cloud mask from HDF5 file using brightness temperature thresholds."""
    try:
        with h5py.File(h5_file, 'r') as hdf:
            img_tir1_temp = hdf['/IMG_TIR1_TEMP'][:]
            img_tir1 = hdf['/IMG_TIR1'][:]
            img_tir1 = img_tir1[0]

            mask = np.ones_like(img_tir1)
            for index in range(img_tir1_temp.shape[0]):
                temp_val = img_tir1_temp[index]
                if temp_val < lower_threshold or temp_val >= upper_threshold:
                    mask[img_tir1 == index] = 0

            return img_tir1, mask
    except KeyError as e:
        print(f"Dataset not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

def load_h5_data(h5_folder, target_size=(256, 256)):
    """Load HDF5 images for model input."""
    images = []
    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5'):
            with h5py.File(os.path.join(h5_folder, filename), 'r') as f:
                img = np.array(f['IMG_TIR1'])
                if img.ndim == 3 and img.shape[0] == 1:
                    img = img[0]
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=-1)
                images.append(img)
    return np.array(images)

def load_masks(mask_folder, target_size=(256, 256)):
    """Load PNG masks for training."""
    masks = []
    for filename in os.listdir(mask_folder):
        if filename.endswith('.png'):
            img = load_img(os.path.join(mask_folder, filename), color_mode='grayscale')
            img = img_to_array(img) / 255.0
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=-1)
            masks.append(img)
    return np.array(masks)
