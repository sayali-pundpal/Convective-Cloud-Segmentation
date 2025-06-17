from sklearn.model_selection import train_test_split
from .data_preprocessing import load_h5_data, load_masks
from .model import deep_unet_model

def train_model(h5_folder, mask_folder):
    """Train the U-Net model on HDF5 images and masks."""
    images = load_h5_data(h5_folder)
    masks = load_masks(mask_folder)

    if len(images) != len(masks):
        raise ValueError("Mismatch between number of images and masks")

    X_train, X_test, y_train, y_test = train_test_split(
        images, masks, test_size=0.2, random_state=42)
    
    model = deep_unet_model(input_shape=X_train[0].shape)
    
    history = model.fit(
        X_train, y_train,
        epochs=1200,
        batch_size=16,
        validation_split=0.2
    )
    
    return model, history
