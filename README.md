# Dog Breed Classifier

This project is a machine-learning application designed to classify dog breeds from images. Using a pre-trained MobileNetV2 model as the backbone, it fine-tunes the network on a dataset of labelled dog images to accurately predict the breed of a given dog. The project demonstrates end-to-end machine learning workflows, including data preprocessing, model training, and evaluation.

## Features

- **Data Preprocessing:** Efficiently processes and batches image data for training and validation.
- **Model Architecture:** Built on a fine-tuned MobileNetV2 model with custom output layers for multi-class classification.
- **Training & Evaluation:** Includes TensorBoard callbacks for real-time monitoring, early stopping for optimal training and various visualizations for model performance.
- **Prediction:** Converts model output probabilities into human-readable labels and visualizes the predictions with confidence levels.

Built with:
<br>
<br>
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
<br>
<br>
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
<br>
<br>
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
<br>
<br>
![Jupyter Notebook](https://img.shields.io/badge/jupyter-fff.svg?style=for-the-badge&logo=jupyter&logoColor=orange)
<br>
<br>
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
<br>
<br>
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
<br>
<br>
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
<br>
<br>
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
<br>
<br>

## 1. Problem Definition
> The goal is to create an end-to-end multi-class image classifier capable of identifying a dog's breed from a given image. The task involves handling unstructured image data and predicting the correct breed out of 120 possible classes.

## 2. Data
The data is downloaded from the Kaggle Bluebook for Bulldozers competition. There are three main datasets:

* `train.zip` is the training set, you are provided with the breed for these dogs.
* `test.zip` the test set, you must predict the probability of each breed for each image.
* `labels.csv` has the breeds for the images in the train set.

### **NOTE**
> You must download a zip file here: https://www.kaggle.com/c/dog-breed-identification/data to run this project locally.
> Create a directory called `data` at the root of the project and place these folders and files into it.
> Setup the directory as such:
> * `data/dog-breed-identification/labels.csv`
> * `data/dog-breed-identification/sample_submission.csv`
> * `data/dog-breed-identification/test/`
> * `data/dog-breed-identification/train/` 

## 3. Evaluation
The model's performance will be evaluated based on prediction probabilities for each dog breed for each test image. 

## 4. Features
* Data Type: Unstructured data (images).
* Number of Classes: 120 dog breeds, representing 120 distinct classes.
* Training Data: Approximately 10,000+ labelled images.
* Test Data: Approximately 10,000+ unlabeled images.
* Approach: Given the unstructured nature of the data, deep learning and transfer learning techniques are employed to build the image classifier.
## 5. Project Steps
The main steps we go through in this project are:

1. **Problem Definition**
* Define the problem of identifying dog breeds from images and determine the scope of the project.
  
2. **Data Collection and Exploration**
Gather the dataset from Kaggle's dog breed identification competition.
Explore the data to understand its structure, including the number of classes, distribution of images, and data quality.

3. **Data Preprocessing**
*Preprocess the images by resizing, normalizing, and augmenting to enhance model generalization.
*Split the data into training, validation, and test sets.

4. **Model Building**
* Build a deep learning model using TensorFlow and Keras.
* Implement transfer learning by leveraging a pre-trained model to improve accuracy and reduce training time.
* Add custom layers to fine-tune the model according to the specific task of dog breed classification.

5. **Model Training**
* Train the model on the processed data, utilizing techniques such as early stopping and learning rate scheduling.
* Monitor the training process using TensorBoard to track performance metrics and ensure model convergence.

## 6. Installation

### 6.1 Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda or Miniconda distribution)
- [Git](https://git-scm.com/)

### 6.2 Installing Dependencies
#### Option 1: Using Conda

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AdrianTomin/dog-vision.git
    cd dog-vision
    ```

2. **Create and activate the Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate dog-vision
    ```

#### Option 2: Using pip

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AdrianTomin/dog-vision.git
    cd dog-vision
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

Choose one of these options to set up the environment, depending on your preference.
`

### 6.3 Setting Up the Environment

1. **Install Jupyter Notebook or JupyterLab:**

    ```bash
    conda install -c conda-forge notebook
    # or for JupyterLab
    conda install -c conda-forge jupyterlab
    ```

2. **Start Jupyter Notebook or JupyterLab:**

    ```bash
    jupyter notebook
    # or for JupyterLab
    jupyter lab
    ```

## 7. Running the Project Locally

1. **Navigate to the project directory:**

    ```bash
    cd dog-vision
    ```

2. **Start the Jupyter Notebook server:**

    ```bash
    jupyter notebook
    ```

3. **Open the notebook:**

    In the Jupyter Notebook interface, open the `dog_vision.ipynb` notebook.

4. **Run the notebook cells:**

    Execute the cells in the notebook to train the model and make predictions. Ensure you have downloaded the dataset and placed it in the appropriate directory as mentioned in the notebook.

---

This README provides a comprehensive guide for setting up the environment, installing dependencies, and running the project locally. Adjust paths and repository links as needed to match your specific setup.


## Badges
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Authors
- [@AdrianTomin](https://www.github.com/AdrianTomin)
