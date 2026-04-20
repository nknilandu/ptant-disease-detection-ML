# Plant Disease Detection System using Deep Learning

This project implements a Convolutional Neural Network (CNN) based system for detecting plant diseases from leaf images. The system is built using **TensorFlow** and **Keras**, with a **Flask** web interface for easy interaction.

## Features

- **Disease Detection**: Identifies diseases in plant leaves using Deep Learning.
- **Web Interface**: A user-friendly Flask-based web application to upload images and get predictions.
- **Model**: A trained CNN model (`model.h5`) that classifies plant diseases.
- **Data Augmentation**: Includes techniques to improve model generalization.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd plant_disease_detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```

2.  **Open the application in your browser:**
    Go to `http://127.0.0.1:5000`

3.  **Upload an image** of a plant leaf and click **"Predict"** to see the detection result.

## Project Structure

```
plant_disease_detection/
├── app.py              # Flask application entry point
├── train.py            # Model training script
├── fix_notebook.py     # Utility to fix notebook issues
├── model.h5            # Trained Keras model
├── Dataset/            # Plant disease dataset
│   ├── Train/
│   └── Test/
├── static/             # Static files (CSS, JS, Images)
│   ├── css/
│   ├── js/
│   └── images/
├── templates/          # HTML templates for the web app
└── requirements.txt    # Project dependencies
```

## Troubleshooting

If you encounter issues with the model training or notebook execution:

- **Dependency Mismatch**: Ensure you are using compatible versions of TensorFlow/Keras (v2.15 recommended) and Python (v3.10 recommended).
- **Notebook Fix**: Run `python fix_notebook.py` to resolve potential import path issues in `Model_Training.ipynb`.
- **Error Logs**: Check the console output for detailed error messages.



