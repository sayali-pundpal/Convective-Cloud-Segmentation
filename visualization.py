import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    # Plot IoU
    ax3.plot(history.history['iou_metric'], label='Training IoU')
    ax3.plot(history.history['val_iou_metric'], label='Validation IoU')
    ax3.set_title('IoU Metric')
    ax3.legend()
    
    plt.tight_layout()
    return fig
