# KYC Form Text Detection Pipeline - Setup & Usage Guide

This guide will help you set up and use the KYC Form Text Detection Pipeline to extract text from Know Your Customer (KYC) forms.

## Prerequisites

1. Python 3.7+ installed
2. Tesseract OCR engine installed
3. Required Python packages

## Installation Steps

### 1. Install Tesseract OCR

#### Windows:
1. Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and note the installation path (default is `C:\Program Files\Tesseract-OCR`)
3. Add Tesseract to your system PATH

#### macOS:
```bash
brew install tesseract
```

#### Linux:
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

### 2. Install Required Python Packages

Create a virtual environment (recommended):
```bash
python -m venv kyc_venv
source kyc_venv/bin/activate  # On Windows: kyc_venv\Scripts\activate
```

Install required packages:
```bash
pip install numpy opencv-python pillow pytesseract pandas scikit-learn tensorflow matplotlib
```

## Project Structure

Create the following directory structure:
```
kyc_form_detection/
├── kyc_form_detector.py       # Main pipeline code
├── training_script.py         # Script for training and evaluation
├── synthetic_forms/           # Will contain generated synthetic forms
├── kyc_dataset/               # Will contain processed training data
│   ├── train/
│   ├── val/
│   └── test/
├── test_forms/                # Place your real KYC forms here
└── results/                   # Output directory for results
```

## Usage Guide

### 1. Generate Synthetic Training Data

Start by generating synthetic KYC forms for training:

```bash
python training_script.py
```

This script will:
- Generate synthetic KYC forms with annotated field positions
- Prepare the training dataset
- Train the form field detector model
- Test the model on real KYC forms
