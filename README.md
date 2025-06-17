# Cloud Masking with U-Net

This project implements a U-Net model for cloud masking in satellite imagery using brightness temperature thresholds and deep learning.

## Project Structure

- `data/`: Contains input HDF5 files and output masks
- `models/`: Saved model weights
- `notebooks/`: Jupyter notebooks for exploration
- `src/`: Source code modules
  - `data_preprocessing.py`: Data loading and preprocessing
  - `model.py`: U-Net model definition
  - `training.py`: Model training script
  - `prediction.py`: Prediction and visualization
  - `visualization.py`: Training metrics visualization

## Installation

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`

## Usage

1. Place your HDF5 files in `data/input/`
2. Generate or place corresponding masks in `data/output/`
3. Run training: `python -m src.training`
4. Make predictions: `python -m src.prediction`
