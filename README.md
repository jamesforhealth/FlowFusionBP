# Branch Contrastive Learning for Precise Cuffless Blood Pressure Prediction using Multi-Modal Data and Multi-Channel Signal on Aurora-BP Dataset

## Project Overview

This project presents a comparative study of different machine learning models for cuffless blood pressure (BP) prediction on the Aurora-BP dataset. We investigate the contributions of various physiological signals, including Photoplethysmography (PPG), Tonometry, and Electrocardiography (ECG), using two primary methodological approaches:

1.  **End-to-End Regression Models**: Directly predict systolic blood pressure (SBP) and diastolic blood pressure (DBP) values from physiological signals.
2.  **Branch Contrastive Learning Model**: A novel approach that first learns a shared representation space by contrasting data points based on their blood pressure similarity, with separate branches specifically designed for SBP and DBP regression targets. This method aims to capture nuanced relationships and learn more effective representations compared to standard contrastive and end-to-end models.

Our findings demonstrate that a fusion model utilizing Tonometry, PPG, and ECG achieves the best prediction performance. Furthermore, the **branch contrastive learning model outperforms traditional end-to-end and standard contrastive learning models**, showcasing its ability to learn effective representations that lead to more accurate blood pressure predictions and meet the precision requirements set by standards such as AAMI and IEEE for blood pressure prediction devices.

## Dataset

This project utilizes the **Aurora-BP** dataset, a comprehensive collection of physiological signals including PPG, Tonometry, and ECG, along with corresponding blood pressure measurements. The dataset provides a valuable resource for researching non-invasive blood pressure monitoring techniques.

## Methodology

We explore and compare the following approaches for cuffless blood pressure prediction:

1.  **End-to-End Regression Models**:
    * We implement direct regression models that take physiological signals (PPG, Tonometry, ECG, or their combinations) as input and directly predict SBP and DBP values.
    * These models may include various architectures such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformer networks tailored for time-series data.

2.  **Contrastive Learning Model**:
    * We implement a contrastive learning framework where the model learns a representation space where data points with similar blood pressure values are closer to each other, and those with dissimilar values are further apart.
    * Regression heads are then attached to this learned representation to predict SBP and DBP.

3.  **Branch Contrastive Learning Model**:
    * Our core contribution is the **Branch Contrastive Learning Model**. This novel architecture extends the standard contrastive learning approach by introducing separate branches within the contrastive learning framework, specifically designed to learn representations relevant to SBP and DBP prediction.
    * This allows the model to capture the distinct underlying physiological factors influencing systolic and diastolic blood pressure, leading to improved performance.

## Input Signals

Our models are designed to handle different combinations of physiological signals:

* **Photoplethysmography (PPG)**
* **Tonometry**
* **Electrocardiography (ECG)**

We evaluate the performance of models trained on individual modalities as well as fusion models that combine these signals. Our results indicate that the **fusion of Tonometry, PPG, and ECG provides the most accurate blood pressure predictions.**

## Key Findings

- The **fusion model utilizing Tonometry, PPG, and ECG achieves the best overall blood pressure prediction accuracy.**
- The **branch contrastive learning model significantly outperforms standard end-to-end regression and traditional contrastive learning models** in predicting both SBP and DBP.
- The **branch contrastive learning approach learns effective representations** that are more accurate in predicting blood pressure, suggesting its potential for improved performance in downstream tasks related to cardiovascular health monitoring.
- The achieved prediction accuracy of our best model **meets the precision requirements outlined in standards such as AAMI and IEEE** for blood pressure prediction devices.

## Evaluation Results

**===== Blood Pressure Prediction Evaluation Results =====**
**Systolic Blood Pressure (SBP)** Mean Absolute Error (MAE): 3.71 mmHg
**Diastolic Blood Pressure (DBP)** Mean Absolute Error (MAE): 2.51 mmHg
**Systolic Blood Pressure (SBP)** Root Mean Squared Error (RMSE): 5.12 mmHg
**Diastolic Blood Pressure (DBP)** Root Mean Squared Error (RMSE): 3.37 mmHg
**Systolic Blood Pressure (SBP)** Coefficient of Determination (R²): 0.8576
**Diastolic Blood Pressure (DBP)** Coefficient of Determination (R²): 0.8109

**===== Error Distribution =====**
Proportion of SBP Error within 5 mmHg: 74.88%
Proportion of DBP Error within 5 mmHg: 88.41%
Proportion of SBP Error within 10 mmHg: 94.57%
Proportion of DBP Error within 10 mmHg: 98.83%
Proportion of SBP Error within 15 mmHg: 98.54%
Proportion of DBP Error within 15 mmHg: 99.78%

## Model Architecture

- `End2EndRegressionModel`: Implements direct regression models for blood pressure prediction.
- `ContrastiveBPModel`: Implements a standard contrastive learning framework for blood pressure prediction.
- `BranchContrastiveBPModel`: Implements our proposed branch contrastive learning model with separate branches for SBP and DBP regression targets.

## Usage

### Training Models

[Instructions on how to train the models would be included here]

## Tools and Technologies

* **Python:** Primary programming language.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical computations.
* **Scikit-learn:** For implementing traditional machine learning models and evaluation metrics.
* **TensorFlow or PyTorch:** For building and training deep learning models, including the contrastive learning frameworks.
* **Libraries for signal processing:** (e.g., SciPy) for potential preprocessing of physiological signals.
* **Matplotlib and Seaborn:** For data visualization and result presentation.

## Potential Future Work

Future research directions may include:

* Further optimization of the branch contrastive learning architecture and training strategies.
* Investigating the interpretability of the learned representations in the branch contrastive model.
* Exploring the generalizability of the proposed approach to other cuffless blood pressure datasets.
* Investigating the application of the learned representations to other downstream tasks related to cardiovascular health.

## Contact Information

James Lin - AI/ML Algorithm Researcher jameslin@flowehealth.com
