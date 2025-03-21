# 3D U-Net for Brain Tumor Segmentation

This repository contains the implementation of pytoch's 3D U-Net model for brain tumor segmentation using the BraTS2020 dataset.

## Installation

## Prerequisites

Ensure that you have the following installed:
- **Python 3.9.19** (This project has been tested with Python 3.9.19. Other versions may work, but compatibility is not guaranteed.)
- pip (Python package installer)

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```bash
    cd 3dunet-glioma
    ```

3. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/MacOS
    venv\Scripts\activate     # For Windows
    ```

4. To install the required dependencies, run:

    ```bash
    pip install -r requirements.txt
    ```

5. Go to folder pytorch-3dunet
    ```bash
    %cd pytorch-3dunet
    ```

6. Install pytorch's 3d unet libraries and model
    ```bash
    pip install .
    ```




## Configuration

Modify config.py according to your dataset path,checkpoint path and training parameters.


## Training

To train and evaluate the model, execute:

```bash
python unet3dmain.py
```
