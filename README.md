
## Pizza-Steak-Sushi Classifier

## Project Overview

This is a deep learning image classifier that identifies whether an image contains **pizza, steak, or sushi**. The model is built using **PyTorch** and utilizes **transfer learning with ResNet18** to achieve high accuracy even with a small dataset.

The dataset for this project was sourced from **David Brooks’ GitHub repository**, and the project was inspired by his tutorial. This demonstrates practical skills in computer vision, deep learning, and Python-based deployment.

---

## Features

* Transfer learning with **ResNet18** pretrained on ImageNet
* Fine-tuning the last convolutional block for better performance
* Dropout to reduce overfitting
* Predict images directly from **URL input** through a CLI
* Clean, organized, and reproducible project structure

---

## Model Performance

* **Best validation accuracy:** 96% (achieved at epoch 4/10)
* **Final model:** saved after peak accuracy
* Handles overfitting using dropout and careful training of frozen layers

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd pizza-steak-sushi-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Activate venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the predictor

```bash
python src/predict.py
```

* Enter an image URL when prompted:

```
Enter image URL: https://upload.wikimedia.org/wikipedia/commons/6/69/Pizza_with_tomatoes.jpg
Predicted class: pizza
```

---

## Project Structure

```
pizza-steak-sushi-classifier/
├── src/
│   ├── data_setup.py       # Handles dataset loading, preprocessing, and dataloaders
│   ├── train.py            # Training script
│   ├── model_builder.py    # Defines the ResNet18 model architecture and modifications
│   ├── engine.py           # Contains training, validation, and metrics functions
│   ├── predict.py          # CLI predictor
│   └── utils.py            # helper function to save model
├── models/                 # Trained model checkpoint (best_model.pth)
├── requirements.txt
└── README.md
```

---

## Dataset

* Sourced from **David Brooks’ GitHub repository**
* Get it from [David Brooks’ GitHub repository](https://github.com/davidbrooks/tutorial-dataset).
* Contains images of pizza, steak, and sushi
* Used for model training and evaluation

---

## Technologies & Skills

* Python & PyTorch
* Transfer learning & fine-tuning
* Image preprocessing & normalization
* Model checkpointing & evaluation
* CLI-based user interaction

---

## Future Improvements

* Add more food categories
* Deploy as a web app using **Streamlit** or **Flask**
* Batch prediction for multiple images
* Enhanced data augmentation for better accuracy

---

## Author

**Sona P Wilson** – AI & Computer Vision Enthusiast | Python & Deep Learning

* GitHub: `https://github.com/SonasWilson`
* Inspired by David Brooks’ tutorial and dataset

---

