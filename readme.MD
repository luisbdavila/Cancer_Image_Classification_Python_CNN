# Breast Cancer Image Classification Using Deep Learning - Convolutional Neural Network (CNN)

## Project Overview
This project leverages deep learning techniques to classify microscopic images of breast tissue into two categories: benign or malignant (binary classification) and into specific cancer types (multiclass classification). Using the BreaKHis dataset, the aim is to support early and accurate breast cancer detection, enhancing diagnosis and treatment planning.

## Objectives
1. Develop a binary classification model to distinguish between benign and malignant breast tissue images.
2. Extend the model to multiclass classification for identifying specific tumor types.
3. Test and optimize various preprocessing techniques, model architectures, and hyperparameters to achieve high classification performance.
4. Analyze and evaluate model performance, highlighting challenges and improvements for future work.

## Problem Statement
Breast cancer is a leading cause of mortality globally, and its early detection is critical for improving survival rates. Traditional diagnostic methods can be time-consuming and prone to error. This project uses deep learning to classify histopathological images, aiming to provide a scalable and accurate diagnostic tool.

## Dataset
- **Source**: [BreaKHis Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- **Images**: High-resolution histopathological images of breast tissue.
- **Metadata**:
  - Path to image.
  - Tumor classification (Benign/Malignant).
  - Specific cancer type (e.g., Adenosis, Ductal Carcinoma).
  - Magnification levels (40x, 100x, 200x, 400x).

## Methodology
### Exploratory Data Analysis
1. Assessed class imbalances in binary and multiclass datasets.
2. Evaluated the impact of magnification levels on classification.
3. Identified and resolved data inconsistencies (e.g., resizing images to a standard dimension of 150x150).

### Preprocessing
1. Tested image transformation techniques, including brightness/contrast adjustments, noise removal, and segmentation.
2. Resized all images to 150x150 while preserving their quality.
3. Stratified dataset splits based on tumor type and magnification levels:
   - Training (64%), Validation (16%), Testing (20%).

### Model Development
1. **Benchmark Model**:
   - Multi-output model using an Inception module for binary and multiclass predictions.
   - Achieved an F1-score of 0.85 for binary classification and 0.38 for multiclass classification.

2. **Optimized Models**:
   - **Binary Classification**:
     - Final model used a simplified Inception-based architecture.
     - Achieved an F1-score of 0.89 on the test set.
   - **Multiclass Classification**:
     - Fine-tuned DenseNet201 pre-trained on ImageNet.
     - Achieved an F1-score of 0.58 on the test set.

### Evaluation
1. Metrics: Precision, recall, F1-score, accuracy.
2. Confusion matrices and classification reports analyzed for error patterns.

### Challenges and Solutions
- Addressed class imbalance using stratification, class weights, and F1-score optimization.
- Fine-tuned pre-trained models to improve multiclass performance.
- Reduced overfitting by limiting model complexity and using cross-validation.

## Results
1. **Binary Classification**:
   - F1-score: 0.89
   - High precision for malignant cases minimizes false positives, reducing unnecessary stress on patients.
   - False negatives remain a challenge, highlighting areas for improvement.
2. **Multiclass Classification**:
   - F1-score: 0.58
   - Effective at identifying distinct classes like Adenosis but struggles with visually similar tumor types.

## Future Work
1. Test larger image sizes (e.g., 300x300) for finer details.
2. Experiment with additional preprocessing methods, such as advanced augmentation techniques.
3. Train or fine-tune models on datasets specific to cancer histopathology.

## Deliverables
1. **Report**: Details methodology, results, and key insights.
2. **Code**:
   - Binary and multiclass classification models.
   - Preprocessing scripts and model training notebooks.
3. **Visualizations**: Confusion matrices, classification reports, and data distribution plots.