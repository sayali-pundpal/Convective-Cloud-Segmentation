import cv2
import numpy as np
import matplotlib.pyplot as plt
from .data_preprocessing import mask_clouds

def predict_and_visualize(model, h5_file):
    """Generate predictions and visualize results."""
    img, bt_mask = mask_clouds(h5_file)
    
    # Preprocess for model input
    img_resized = cv2.resize(img, (256, 256))
    img_resized = np.expand_dims(img_resized, axis=-1)
    
    # Predict
    predicted_mask = model.predict(np.expand_dims(img_resized, axis=0))
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(bt_mask, cmap='gray')
    axes[1].set_title('Mask (BT Thresholds)')
    axes[1].axis('off')

    axes[2].imshow(predicted_mask.squeeze(), cmap='gray')
    axes[2].set_title('Predicted Mask (U-Net)')
    axes[2].axis('off')

    plt.tight_layout()
    return fig
