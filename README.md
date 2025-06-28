
# Flower Species Classifier – TensorFlow Image Classification Project

This project is part of the **Udacity Intro to Machine Learning with TensorFlow Nanodegree**. It demonstrates how to build a deep learning model to classify images of flowers into one of 102 categories using the **Oxford Flowers 102** dataset.

## Project Overview

Artificial Intelligence is becoming a core part of many applications, and image classification is one of its most common use cases. In this project, a pre-trained convolutional neural network (MobileNetV2) is used to extract features from flower images. A custom classifier is trained on top of it to identify flower species.

The model can be used in mobile apps, websites, or any system where identifying flowers from images is required.

## Project Steps

1. **Load and Preprocess Data**
   Load the Oxford 102 Flowers dataset using `tensorflow_datasets`, normalize the pixel values, and resize images to the expected input size.

2. **Build and Train the Model**
   Load the MobileNetV2 model from TensorFlow Hub with pre-trained weights. Add a dense classifier and train it using the flower dataset.

3. **Evaluate Performance**
   Test the trained model on a separate test set to check how well it generalizes.

4. **Make Predictions**
   Write a `predict()` function that takes an image and returns the top K most likely classes.

5. **Visualize Predictions**
   Use matplotlib to display the image alongside the predicted class probabilities.

## Dataset

* Name: [Oxford Flowers 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* Number of classes: 102 flower categories
* Image count: Over 8,000 images
* Provided splits: Training, Validation, and Test

The images are resized to 224x224 to match the MobileNetV2 input size, and normalized to have values between 0 and 1.

## Features

* **Transfer Learning with MobileNetV2**
* **Custom classifier trained using TensorFlow**
* **Model saved as HDF5 (.h5) for reuse**
* **Prediction script with top-k inference support**
* **Visualization of prediction results**

## Installation Instructions

1. Clone the repository:

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

4. Train the model and make predictions by running cells in order.

## Project Structure

```
flower-classifier/
├── Project_Image_Classifier_Project.ipynb     # Jupyter notebook with training pipeline
├── flower_classifier.hdf5                     # Trained model
├── label_map.json                             # Flower label to name mapping
├── test_images/                               # Sample images for testing
├── predict.py                                 # Script for running inference from CLI
├── README.md                                  # Project documentation
```

## Example Usage

Use the `predict.py` script to classify a flower image:

```python
from predict import predict

image_path = './test_images/wild_pansy.jpg'
model_path = './flower_classifier.hdf5'

probs, classes = predict(image_path, model_path, top_k=5)
print(probs)
print(classes)
```

## Results

* Validation accuracy: \~75%
* Test accuracy: \~71%
* Trained using 10 epochs with MobileNetV2 as a frozen feature extractor

## Future Improvements

* Fine-tune the MobileNetV2 model for better accuracy
* Add data augmentation for robustness
* Deploy the model as a mobile or web application
* Extend to other classification domains (e.g., vehicles, animals)
