# Superpixel Mean-Color Regression using U-Net

This project trains a U-Net to learn the mean-color representation of SLIC superpixels using supervised regression.

## Structure

- `model/`: Contains the U-Net model
- `slic/`: Contains SLIC preprocessing functions
- `utils/`: Visualization utilities
- `train.py`: Main training script
- `test.py` : Testing script

## Setup

```bash
pip install -r requirements.txt
python train.py
python test.py
