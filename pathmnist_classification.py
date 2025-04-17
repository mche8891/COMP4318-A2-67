#!/usr/bin/env python3
# PathMNIST Classification
# COMP4318 Assignment 2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import pandas as pd
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Data loading
def load_data():
    print("Loading data...")
    X_train = np.load('Assignment2Data/X_train.npy')
    y_train = np.load('Assignment2Data/y_train.npy')
    X_test = np.load('Assignment2Data/X_test.npy')
    y_test = np.load('Assignment2Data/y_test.npy')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

# Data exploration
def explore_data(X_train, y_train, X_test, y_test):
    print("\n===== Data Exploration =====")
    
    # Check for any missing values
    print(f"Missing values in X_train: {np.isnan(X_train).any()}")
    print(f"Missing values in y_train: {np.isnan(y_train).any()}")
    
    # Class distribution
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Classes: {unique_classes}")
    print(f"Class distribution: {class_counts}")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=unique_classes, y=class_counts)
    plt.title('Class Distribution in Training Data')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Visualize some images
    plt.figure(figsize=(15, 10))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        if len(X_train.shape) == 4:  # If images are color (assuming NHWC format)
            plt.imshow(X_train[i])
        else:  # If images are grayscale
            plt.imshow(X_train[i], cmap='gray')
        plt.title(f'Class: {y_train[i]}')
        plt.axis('off')
    plt.savefig('sample_images.png')
    plt.close()
    
    # Check image dimensions and pixel intensity range
    print(f"Image dimensions: {X_train.shape[1:] if len(X_train.shape) > 1 else 'Not image data'}")
    print(f"Min pixel value: {X_train.min()}")
    print(f"Max pixel value: {X_train.max()}")
    print(f"Mean pixel value: {X_train.mean()}")
    print(f"Std pixel value: {X_train.std()}")
    
    # Plot pixel intensity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(X_train.flatten(), bins=50)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig('pixel_distribution.png')
    plt.close()
    
    # Show mean images for each class
    plt.figure(figsize=(15, 10))
    for i, class_idx in enumerate(unique_classes):
        if i >= 10:  # Limit to first 10 classes if there are many
            break
        class_mask = (y_train == class_idx)
        if class_mask.sum() > 0:
            mean_img = X_train[class_mask].mean(axis=0)
            plt.subplot(2, 5, i+1)
            if len(mean_img.shape) == 3:  # Color image
                plt.imshow(mean_img.astype(np.uint8))
            else:  # Grayscale
                plt.imshow(mean_img, cmap='gray')
            plt.title(f'Mean of Class {class_idx}')
            plt.axis('off')
    plt.savefig('class_means.png')
    plt.close()

# Data preprocessing
def preprocess_data(X_train, y_train, X_test, y_test):
    print("\n===== Data Preprocessing =====")
    
    # Store original data for Random Forest (will be flattened later)
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    
    # Reshape and normalize images if needed
    # Check if images are already in the right format
    if len(X_train.shape) == 4:
        print("Images already have channel dimension")
    else:
        # Reshape if needed (e.g., for grayscale)
        print("Reshaping images to add channel dimension")
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    print(f"Normalized pixel range: [{X_train.min()}, {X_train.max()}]")
    
    # For Random Forest, flatten the images
    X_train_rf = X_train_original.reshape(X_train_original.shape[0], -1).astype('float32') / 255.0
    X_test_rf = X_test_original.reshape(X_test_original.shape[0], -1).astype('float32') / 255.0
    print(f"Random Forest input shape: {X_train_rf.shape}")
    
    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Split training data to create a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_cat, test_size=0.2, random_state=42, stratify=y_train_cat
    )
    
    # Also split the RF training data
    X_train_rf_split, X_val_rf, y_train_rf_split, y_val_rf = train_test_split(
        X_train_rf, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set shape: {X_train_split.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Random Forest training set shape: {X_train_rf_split.shape}")
    
    return X_train_split, X_val, X_test, y_train_split, y_val, y_test_cat, num_classes, X_train_rf_split, X_val_rf, X_test_rf, y_train_rf_split, y_val_rf, y_test

# MLP model
def create_mlp_model(input_shape, num_classes, learning_rate=0.001, dropout_rate=0.5, hidden_units=512):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(hidden_units, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(hidden_units // 2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(hidden_units // 4, activation='relu'),
        BatchNormalization(),
        Dropout(min(dropout_rate * 0.6, 0.7)),  # 避免dropout率过高
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# CNN model
def create_cnn_model(input_shape, num_classes, learning_rate=0.001, dropout_rate=0.25, filters=32):
    model = Sequential([
        # First convolutional block
        Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        # Second convolutional block
        Conv2D(filters * 2, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters * 2, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        # Third convolutional block
        Conv2D(filters * 4, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters * 4, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        # Fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(min(dropout_rate * 1.5, 0.8)),  # 确保dropout率不超过0.8
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Random Forest model
def create_and_train_rf(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, min_samples_split=2):
    print("\n===== Training Random Forest =====")
    start_time = time.time()
    
    # Create and train model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('RF_confusion_matrix.png')
    plt.close()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report for Random Forest:")
    print(report_df)
    
    # Feature importance
    if max_depth is not None:  # Only plot if we have a reasonable number of features
        n_features = min(20, X_train.shape[1])  # Limit to top 20 features
        feature_importance = pd.Series(rf_model.feature_importances_, 
                                       index=[f"pixel_{i}" for i in range(X_train.shape[1])])
        plt.figure(figsize=(10, 8))
        feature_importance.nlargest(n_features).plot(kind='barh')
        plt.title('Random Forest Feature Importance')
        plt.savefig('RF_feature_importance.png')
        plt.close()
    
    return rf_model, accuracy, report_df, training_time

# Training and evaluation function for neural networks
def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, batch_size=None, epochs=30):
    print(f"\n===== Training {model_name} =====")
    
    # 根据可用GPU数量增加批量大小
    if batch_size is None:
        gpu_count = len(tf.config.list_physical_devices('GPU'))
        base_batch_size = 64
        batch_size = base_batch_size * max(1, min(gpu_count, 4))  # 根据GPU数量调整批量大小，但不超过4倍
    
    print(f"Using batch size: {batch_size}")
    
    start_time = time.time()
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )
    
    # Train the model with progress bar
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1  # 显示进度条
    )
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate on test set with progress bar
    print(f"\n===== Evaluating {model_name} =====")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)  # 显示评估进度条
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png')
    plt.close()
    
    # Get predictions and create confusion matrix
    y_pred = model.predict(X_test, verbose=1)  # 添加预测进度条
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(f"\nClassification Report for {model_name}:")
    print(report_df)
    
    return test_accuracy, report_df, training_time, history

# Hyperparameter tuning for Random Forest
def tune_random_forest(X_train, y_train, X_val, y_val):
    print("\n===== Hyperparameter Tuning for Random Forest =====")
    
    # For faster tuning, sample a subset of data if dataset is large
    if X_train.shape[0] > 10000:
        print("Sampling subset of data for Random Forest tuning...")
        sample_size = min(10000, X_train.shape[0])
        indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create a base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)  # 使用所有核心
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,  # 使用所有核心
        verbose=2,  # 增加详细程度
        scoring='accuracy'
    )
    
    start_time = time.time()
    grid_search.fit(X_train_sample, y_train_sample)
    tuning_time = time.time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    # Validate on the validation set
    best_rf = grid_search.best_estimator_
    val_accuracy = best_rf.score(X_val, y_val)
    print(f"Validation accuracy with best parameters: {val_accuracy:.4f}")
    
    # Create a results DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Plot the results of different hyperparameter combinations
    plt.figure(figsize=(15, 10))
    
    # Plot effect of n_estimators
    plt.subplot(2, 2, 1)
    sns.boxplot(x='param_n_estimators', y='mean_test_score', data=results)
    plt.title('Effect of n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    
    # Plot effect of max_depth
    plt.subplot(2, 2, 2)
    results_no_none = results[results['param_max_depth'].notna()]
    if not results_no_none.empty:
        sns.boxplot(x='param_max_depth', y='mean_test_score', data=results_no_none)
    plt.title('Effect of max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    
    # Plot effect of min_samples_split
    plt.subplot(2, 2, 3)
    sns.boxplot(x='param_min_samples_split', y='mean_test_score', data=results)
    plt.title('Effect of min_samples_split')
    plt.xlabel('min_samples_split')
    plt.ylabel('Accuracy')
    
    # Plot top combinations
    plt.subplot(2, 2, 4)
    top_results = results.sort_values('mean_test_score', ascending=False).head(10)
    sns.barplot(x=range(len(top_results)), y='mean_test_score', data=top_results)
    plt.title('Top 10 Hyperparameter Combinations')
    plt.xlabel('Combination Index')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('RF_hyperparameter_tuning.png')
    plt.close()
    
    return grid_search.best_params_

# Hyperparameter tuning for MLP
def tune_mlp(X_train, y_train, X_val, y_val, input_shape, num_classes):
    print("\n===== Hyperparameter Tuning for MLP =====")
    
    # 增加GPU存在时的批量大小
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    base_batch_size = 64
    batch_size = base_batch_size * max(1, min(gpu_count, 4))  # 根据GPU数量调整批量大小，但不超过4倍
    print(f"Using batch size: {batch_size} (detected {gpu_count} GPUs)")
    
    # Define hyperparameter combinations to try
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rates = [0.3, 0.5, 0.7]
    hidden_units = [128, 256, 512]
    
    # Track results
    results = []
    best_val_accuracy = 0
    best_params = {}
    
    total_combinations = len(learning_rates) * len(dropout_rates) * len(hidden_units)
    print(f"Testing {total_combinations} combinations...")
    
    start_time = time.time()
    
    # Try different combinations
    for lr in learning_rates:
        for dropout in dropout_rates:
            for units in hidden_units:
                print(f"\nTrying: lr={lr}, dropout={dropout}, hidden_units={units}")
                
                # Create and compile model
                model = Sequential([
                    Flatten(input_shape=input_shape),
                    Dense(units, activation='relu'),
                    BatchNormalization(),
                    Dropout(dropout),
                    Dense(units // 2, activation='relu'),
                    BatchNormalization(),
                    Dropout(dropout),
                    Dense(num_classes, activation='softmax')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # Use adjusted batch size
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=10,  # Limit epochs for tuning
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=1  # 显示进度条
                )
                
                # Evaluate
                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)  # 显示验证评估进度条
                print(f"Validation accuracy: {val_accuracy:.4f}")
                
                # Track results
                results.append({
                    'learning_rate': lr,
                    'dropout_rate': dropout,
                    'hidden_units': units,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'epochs_trained': len(history.history['loss'])
                })
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params = {
                        'learning_rate': lr,
                        'dropout_rate': dropout,
                        'hidden_units': units
                    }
    
    tuning_time = time.time() - start_time
    print(f"Tuning completed in {tuning_time:.2f} seconds")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Visualize hyperparameter effects
    plt.figure(figsize=(15, 10))
    
    # Effect of learning rate
    plt.subplot(2, 2, 1)
    sns.boxplot(x='learning_rate', y='val_accuracy', data=results_df)
    plt.title('Effect of Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    
    # Effect of dropout rate
    plt.subplot(2, 2, 2)
    sns.boxplot(x='dropout_rate', y='val_accuracy', data=results_df)
    plt.title('Effect of Dropout Rate')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Accuracy')
    
    # Effect of hidden units
    plt.subplot(2, 2, 3)
    sns.boxplot(x='hidden_units', y='val_accuracy', data=results_df)
    plt.title('Effect of Hidden Units')
    plt.xlabel('Hidden Units')
    plt.ylabel('Validation Accuracy')
    
    # Top combinations
    plt.subplot(2, 2, 4)
    top_results = results_df.sort_values('val_accuracy', ascending=False).head(10)
    plt.bar(range(len(top_results)), top_results['val_accuracy'])
    plt.title('Top 10 Hyperparameter Combinations')
    plt.xlabel('Combination Index')
    plt.ylabel('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('MLP_hyperparameter_tuning.png')
    plt.close()
    
    print("\nAll tuning results:")
    print(results_df.sort_values('val_accuracy', ascending=False).head(10))
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return best_params

# Hyperparameter tuning for CNN
def tune_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes):
    print("\n===== Hyperparameter Tuning for CNN =====")
    
    # 增加GPU存在时的批量大小
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    base_batch_size = 64
    batch_size = base_batch_size * max(1, min(gpu_count, 4))  # 根据GPU数量调整批量大小，但不超过4倍
    print(f"Using batch size: {batch_size} (detected {gpu_count} GPUs)")
    
    # Define hyperparameter combinations to try
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rates = [0.25, 0.4, 0.5]
    filters = [16, 32, 64]
    
    # Track results
    results = []
    best_val_accuracy = 0
    best_params = {}
    
    total_combinations = len(learning_rates) * len(dropout_rates) * len(filters)
    print(f"Testing {total_combinations} combinations...")
    
    start_time = time.time()
    
    # Try different combinations
    for lr in learning_rates:
        for dropout in dropout_rates:
            for filter_size in filters:
                print(f"\nTrying: lr={lr}, dropout={dropout}, filters={filter_size}")
                
                # Create model
                model = Sequential([
                    # First convolutional block
                    Conv2D(filter_size, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
                    BatchNormalization(),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(dropout),
                    
                    # Second convolutional block
                    Conv2D(filter_size * 2, kernel_size=(3, 3), activation='relu', padding='same'),
                    BatchNormalization(),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(dropout),
                    
                    # Fully connected layers
                    Flatten(),
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dropout(min(dropout * 1.5, 0.8)),  # 确保dropout率不超过0.8
                    Dense(num_classes, activation='softmax')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,  # 使用根据GPU数量调整的批量大小
                    epochs=10,  # Reduced for tuning
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=1  # 显示进度条
                )
                
                # Evaluate
                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)  # 显示验证评估进度条
                print(f"Validation accuracy: {val_accuracy:.4f}")
                
                results.append({
                    'learning_rate': lr,
                    'dropout_rate': dropout,
                    'filters': filter_size,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'epochs_trained': len(history.history['loss'])
                })
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params = {
                        'learning_rate': lr,
                        'dropout_rate': dropout,
                        'filters': filter_size
                    }
    
    tuning_time = time.time() - start_time
    print(f"Tuning completed in {tuning_time:.2f} seconds")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Visualize hyperparameter effects
    plt.figure(figsize=(15, 10))
    
    # Effect of learning rate
    plt.subplot(2, 2, 1)
    sns.boxplot(x='learning_rate', y='val_accuracy', data=results_df)
    plt.title('Effect of Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    
    # Effect of dropout rate
    plt.subplot(2, 2, 2)
    sns.boxplot(x='dropout_rate', y='val_accuracy', data=results_df)
    plt.title('Effect of Dropout Rate')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Accuracy')
    
    # Effect of filters
    plt.subplot(2, 2, 3)
    sns.boxplot(x='filters', y='val_accuracy', data=results_df)
    plt.title('Effect of Initial Filters')
    plt.xlabel('Number of Filters')
    plt.ylabel('Validation Accuracy')
    
    # Top combinations
    plt.subplot(2, 2, 4)
    top_results = results_df.sort_values('val_accuracy', ascending=False).head(10)
    plt.bar(range(len(top_results)), top_results['val_accuracy'])
    plt.title('Top 10 Hyperparameter Combinations')
    plt.xlabel('Combination Index')
    plt.ylabel('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('CNN_hyperparameter_tuning.png')
    plt.close()
    
    print("\nAll tuning results:")
    print(results_df.sort_values('val_accuracy', ascending=False).head(10))
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return best_params

# Model comparison function
def compare_models(model_results):
    print("\n===== Model Comparison =====")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(model_results)
    print(comparison_df)
    
    # Accuracy comparison
    plt.figure(figsize=(12, 10))
    
    # Accuracy plot
    plt.subplot(2, 1, 1)
    sns.barplot(x='model', y='accuracy', data=comparison_df)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    
    # Training time plot
    plt.subplot(2, 1, 2)
    sns.barplot(x='model', y='training_time', data=comparison_df)
    plt.title('Model Training Time Comparison')
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    return comparison_df

# Main function
def main():
    # 检测并显示可用GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Available GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f" - {gpu.name}")
        
        # 设置GPU内存增长
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {gpu.name}")
            except:
                print(f"Memory growth setting failed for {gpu.name}")
    else:
        print("No GPUs detected, using CPU")
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Explore data
    explore_data(X_train, y_train, X_test, y_test)
    
    # Preprocess data
    (X_train_nn, X_val_nn, X_test_nn, 
     y_train_nn, y_val_nn, y_test_nn, num_classes,
     X_train_rf, X_val_rf, X_test_rf,
     y_train_rf, y_val_rf, y_test_rf) = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Get input shape for models
    input_shape = X_train_nn.shape[1:]
    print(f"Input shape for NN models: {input_shape}")
    
    # Whether to perform hyperparameter tuning
    do_hyperparameter_tuning = True
    
    # Results tracker
    model_results = []
    
    # # ------- RANDOM FOREST -------
    # if do_hyperparameter_tuning:
    #     print("\nTuning Random Forest hyperparameters...")
    #     rf_best_params = tune_random_forest(X_train_rf, y_train_rf, X_val_rf, y_val_rf)
    #     rf_model, rf_accuracy, rf_report, rf_time = create_and_train_rf(
    #         X_train_rf, y_train_rf, X_test_rf, y_test_rf,
    #         n_estimators=rf_best_params['n_estimators'],
    #         max_depth=rf_best_params['max_depth'],
    #         min_samples_split=rf_best_params['min_samples_split']
    #     )
    # else:
    #     # Create and train with default parameters
    #     rf_model, rf_accuracy, rf_report, rf_time = create_and_train_rf(
    #         X_train_rf, y_train_rf, X_test_rf, y_test_rf
    #     )
    
    # model_results.append({
    #     'model': 'Random Forest',
    #     'accuracy': rf_accuracy,
    #     'training_time': rf_time
    # })
    
    # ------- MLP MODEL -------
    if do_hyperparameter_tuning:
        print("\nTuning MLP hyperparameters...")
        mlp_best_params = tune_mlp(X_train_nn, y_train_nn, X_val_nn, y_val_nn, input_shape, num_classes)
        mlp_model = create_mlp_model(
            input_shape, 
            num_classes,
            learning_rate=mlp_best_params['learning_rate'],
            dropout_rate=mlp_best_params['dropout_rate'],
            hidden_units=mlp_best_params['hidden_units']
        )
    else:
        # Create with default parameters
        mlp_model = create_mlp_model(input_shape, num_classes)
    
    # Train and evaluate MLP
    mlp_accuracy, mlp_report, mlp_time, mlp_history = train_and_evaluate(
        mlp_model, 
        X_train_nn, 
        y_train_nn, 
        X_val_nn, 
        y_val_nn, 
        X_test_nn, 
        y_test_nn, 
        'MLP'
    )
    
    model_results.append({
        'model': 'MLP',
        'accuracy': mlp_accuracy,
        'training_time': mlp_time
    })
    
    # ------- CNN MODEL -------
    # if do_hyperparameter_tuning:
    #     print("\nTuning CNN hyperparameters...")
    #     cnn_best_params = tune_cnn(X_train_nn, y_train_nn, X_val_nn, y_val_nn, input_shape, num_classes)
    #     cnn_model = create_cnn_model(
    #         input_shape,
    #         num_classes,
    #         learning_rate=cnn_best_params['learning_rate'],
    #         dropout_rate=cnn_best_params['dropout_rate'],
    #         filters=cnn_best_params['filters']
    #     )
    # else:
    #     # Create with default parameters
    #     cnn_model = create_cnn_model(input_shape, num_classes)
    
    # # Train and evaluate CNN
    # cnn_accuracy, cnn_report, cnn_time, cnn_history = train_and_evaluate(
    #     cnn_model, 
    #     X_train_nn, 
    #     y_train_nn, 
    #     X_val_nn, 
    #     y_val_nn, 
    #     X_test_nn, 
    #     y_test_nn, 
    #     'CNN'
    # )
    
    # model_results.append({
    #     'model': 'CNN',
    #     'accuracy': cnn_accuracy,
    #     'training_time': cnn_time
    # })
    
    # # Compare all models
    # comparison_df = compare_models(model_results)
    
    # print("\nTask completed. Check the saved images for visualizations.")

if __name__ == "__main__":
    main() 