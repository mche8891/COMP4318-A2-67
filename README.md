# PathMNIST Image Classification

This project implements and compares machine learning algorithms for classifying PathMNIST images, which contain 28x28 color images of microscope slides of normal and abnormal body tissues.

## Requirements

The following libraries are required:
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- pandas

You can install them using:
```
pip install numpy matplotlib seaborn scikit-learn tensorflow pandas
```

## Dataset

The PathMNIST dataset is provided in the `Assignment2Data` folder:
- `X_train.npy` - Training images
- `y_train.npy` - Training labels
- `X_test.npy` - Test images
- `y_test.npy` - Test labels

## Running the Code

To run the complete script:
```
python pathmnist_classification.py
```

## What the Code Does

1. **Data Loading**: Loads the PathMNIST dataset from .npy files
2. **Data Exploration**: Analyzes dataset properties, class distribution, and visualizes sample images
3. **Data Preprocessing**: Normalizes pixel values, one-hot encodes labels, and creates a validation set
4. **Model Implementation**:
   - Random Forest Classifier (traditional ML approach)
   - Multilayer Perceptron (MLP)
   - Convolutional Neural Network (CNN)
5. **Hyperparameter Tuning**:
   - Random Forest: n_estimators, max_depth, min_samples_split
   - MLP: learning rate, dropout rate, hidden units
   - CNN: learning rate, dropout rate, filters
6. **Training and Evaluation**: Trains all models and evaluates them on test data
7. **Model Comparison**: Compares model performance and visualizes results

## Output

The script generates several visualization files:
- **Data Exploration**:
  - `class_distribution.png` - Distribution of classes in the training set
  - `sample_images.png` - Sample images from the dataset
  - `pixel_distribution.png` - Distribution of pixel intensities
  - `class_means.png` - Mean images for each class

- **Hyperparameter Tuning**:
  - `RF_hyperparameter_tuning.png` - Random Forest tuning results
  - `MLP_hyperparameter_tuning.png` - MLP tuning results
  - `CNN_hyperparameter_tuning.png` - CNN tuning results

- **Model Results**:
  - `RF_confusion_matrix.png` - Random Forest confusion matrix
  - `RF_feature_importance.png` - Random Forest feature importance
  - `MLP_history.png` - MLP training history
  - `MLP_confusion_matrix.png` - MLP confusion matrix
  - `CNN_history.png` - CNN training history
  - `CNN_confusion_matrix.png` - CNN confusion matrix
  - `model_comparison.png` - Comparison of all model accuracies and training times

The script also outputs detailed information about the dataset, hyperparameter tuning, and model performance to the console.