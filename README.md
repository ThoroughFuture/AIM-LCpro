# AIM-LCpro: Accurate Prediction of Disease-Free and Overall Survival in Non-Small Cell Lung Cancer Using Patient-Level Multimodal Weakly Supervised Learning

## Overview

AIM-LCpro focuses on developing a multimodal AI model that integrates whole-slide images (WSIs) and dense clinical data to predict disease-free survival (DFS) and overall survival (OS) with high accuracy for non-small cell lung cancer (NSCLC) patients undergoing surgery.

## Potential Impact

This model has the potential to revolutionize postoperative decision-making by providing clinicians with a precise tool for predicting DFS and OS, thereby improving patient outcomes.

## Contents

1. **image_feature_extract.py**
   - **Description:** This script utilizes a pre-trained ResNet50 model to extract features from images and saves these features to a specified directory.

2. **processing_features.py**
   - **Description:** This script is responsible for reading and preprocessing clinical data, image features, and preparing the data for training machine learning models.

3. **feature_train_mlp.py**
   - **Description:** This script encompasses a complete training and evaluation process. It separates predictions and calculates and plots ROC curves during the training process.

4. **threshold.py**
   - **Description:** The script employs a two-threshold voting strategy for classifying model predictions and evaluating their performance.

5. **survival_prediction.py**
   - **Description:** This script functions as a patient survival predictor.

## Usage

Each script is designed to be run independently. Ensure you have the necessary libraries installed before running any of the scripts.

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas

### Installation

To install the required libraries, run:

```
pip install tensorflow keras numpy matplotlib pandas
````

### Running the Scripts

1.  **Extracting Image Features:**
    ```
    python image_feature_extract.py
    ```

2.  **Processing Features:**
    ```
    python processing_features.py
    ```

3.  **Training and Evaluating the Model:**
    ```
    python feature_train_mlp.py
    ```

4.  **Applying Threshold Strategy:**
    ```
    python threshold.py
    ```

5.  **Predicting Survival:**
    ```
    python survival_prediction.py
    ```

## License

This project is open source and available under the MIT License.
