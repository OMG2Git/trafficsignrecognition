# Traffic Sign Recognition

This project aims to recognize and classify traffic signs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The system detects and classifies traffic signs in real-time using a webcam or DroidCam feed.

## Project Overview

- **Dataset**: The dataset used for training contains images of various traffic signs (e.g., speed limits, right-turn signs).
- **Model**: A CNN model is trained to classify these traffic signs, and it can be fine-tuned to improve accuracy.
- **Real-Time Prediction**: The project includes functionality to predict traffic signs in real time using a webcam or DroidCam.
- **Tools**: The project uses **TensorFlow**, **OpenCV**, and **Roboflow** for training, object detection, and real-time predictions.

---

## Requirements

### 1. Install Required Libraries

Use the following command to install all the dependencies required for the project:


pip install -r requirements.txt
Here is the list of libraries included in requirements.txt:

tensorflow
opencv-python
numpy
pyttsx3
roboflow
queue
tensorflow-keras
matplotlib

### 2. Setup Roboflow API Key

This project uses Roboflow for object detection to recognize traffic signs. To use Roboflow, you'll need to create an account and obtain an API key.

Steps to Set Up the Roboflow API Key:
Go to Roboflow and create an account.
Create a new project for your traffic sign dataset.
After creating the project, navigate to your account settings to get the API key.
In the Python scripts (such as real_time_prediction.py, fine_tune_model.py), replace the placeholder YOUR_ROBOFLOW_API_KEY with your actual Roboflow API Key.
Example code:


Copy code
```bash
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
Also, update the Roboflow workspace and project version in the script as follows:
```
Copy code
```bash
detection_model = rf.workspace("om").project("tabela_v1.2").version(12).model
```
### 3. Dataset Path

The paths for the train and test directories need to be set correctly based on where your dataset is stored locally.

Update the following paths in your scripts:
python
Copy code
```bash
train_dir = r"C:\\Users\\your_user\\path\\to\\Train"
test_dir = r"C:\\Users\\your_user\\path\\to\\Test"
```
Ensure your dataset is structured as follows:

```bash
- Train/
  - 30/
  - 50/
  - 60/
  - 70/
  - 80/
  - right/
- Test/
  - 30/
  - 50/
  - 60/
  - 70/
  - 80/
  - right/
```

### 4. Dataset Overview

The dataset used in this project contains images of traffic signs categorized into multiple classes, such as:

Speed limit signs (30, 50, 60, 70, 80)
Right-turn signs

### 5. Model Architecture

The model used for traffic sign classification is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. It contains the following components:

3 Convolutional layers with MaxPooling layers for feature extraction.
A Flatten layer to convert 2D data into 1D vectors.
A Dense fully connected layer for classification.
A Dropout layer for regularization to prevent overfitting.
A Softmax output layer for multi-class classification.
Once trained, the model is saved as train3.h5 for deployment.

## Usage

### 1. Real-Time Prediction

To use the system for real-time prediction, connect your webcam or DroidCam, and run the following script:


Copy code
```bash
python real_time_prediction.py
```
This will:

Use your webcam or DroidCam feed to detect and classify traffic signs.
Display the live video feed with bounding boxes around detected traffic signs.
Announce the predicted traffic sign using text-to-speech.
Make sure you have set the correct Roboflow API key and dataset paths before running the script. 

### 2. Model Training and Evaluation

Training from Scratch
To train the model from scratch using your dataset, run:


Copy code
```bash
python train.py
```
This will:

Load the images from the Train and Test directories.
Preprocess the images (resize, normalize).
Train the model with the specified parameters (batch size, number of epochs).
Save the trained model as train3.h5.
Fine-Tuning a Pre-Trained Model
To fine-tune a pre-trained model, use the following script:


Copy code
```bash
python fine_tune_model.py
```
This will:

Load the pre-trained model (train3.h5).
Freeze the layers of the pre-trained model.
Add new dense layers for fine-tuning.
Continue training the model with a smaller learning rate.
Save the fine-tuned model as finetuned_train3_v2.keras.

## Troubleshooting
Missing Dependencies: Make sure all dependencies in requirements.txt are installed by running:


Copy code
```bash
pip install -r requirements.txt
```
Roboflow API Key: Ensure that the correct API key and project details are used in the scripts.

Dataset Paths: Verify that the dataset is structured correctly and the paths are set in the scripts.

## Authors
Om Gosavi: Developer and Contributor

## License
This project is licensed under the MIT License - see the LICENSE file for details.