import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import os

# Configure GPU memory usage or disable GPU if there are issues
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # Limit GPU memory growth - prevents TF from taking all GPU memory at once
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled")
    except:
        # If the above fails, try disabling GPU
        print("Could not set memory growth, disabling GPU")
        tf.config.set_visible_devices([], 'GPU')

# For complete GPU disabling, uncomment the line below
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data exploration
def explore_data(X_train, y_train):
    print("\n=== Data Exploration ===")
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    num_classes = len(unique_classes)
    print(f"Number of classes: {num_classes}")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(unique_classes, class_counts)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(unique_classes)
    plt.savefig('class_distribution.png')
    
    # Display sample images from each class
    plt.figure(figsize=(15, 15))
    for i, class_idx in enumerate(unique_classes):
        if i >= 9:  # Limit to maximum 9 classes for display
            break
        class_samples = X_train[y_train == class_idx]
        for j in range(min(5, len(class_samples))):
            plt.subplot(num_classes, 5, i*5 + j + 1)
            plt.imshow(class_samples[j])
            plt.title(f'Class {class_idx}')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    
    # Check for pixel intensity distribution
    plt.figure(figsize=(12, 6))
    plt.hist(X_train.ravel(), bins=50)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig('pixel_distribution.png')
    
    # Calculate mean and standard deviation
    mean_per_class = []
    std_per_class = []
    for class_idx in unique_classes:
        class_samples = X_train[y_train == class_idx]
        mean_per_class.append(np.mean(class_samples))
        std_per_class.append(np.std(class_samples))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(unique_classes, mean_per_class)
    plt.title('Mean Pixel Value by Class')
    plt.xlabel('Class')
    plt.ylabel('Mean Pixel Value')
    
    plt.subplot(1, 2, 2)
    plt.bar(unique_classes, std_per_class)
    plt.title('Standard Deviation of Pixel Values by Class')
    plt.xlabel('Class')
    plt.ylabel('Standard Deviation')
    plt.tight_layout()
    plt.savefig('pixel_stats_by_class.png')
    
    return num_classes

# Data preprocessing
def preprocess_data(X_train, X_test, y_train, y_test, num_classes):
    print("\n=== Data Preprocessing ===")
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Split training data further into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # One-hot encode the labels
    y_train_cat = to_categorical(y_train_split, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"After preprocessing:")
    print(f"X_train shape: {X_train_split.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train_cat.shape}")
    print(f"y_val shape: {y_val_cat.shape}")
    
    return X_train_split, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train_split, y_val, y_test

# Build MLP model
def build_mlp(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train and evaluate models
def train_and_evaluate(model, model_name, X_train, X_val, y_train, y_val, X_test, y_test, y_test_orig):
    print(f"\n=== Training {model_name} ===")
    
    # Define callbacks for training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print(f"\n{model_name} - Classification Report:")
    print(classification_report(y_test_orig, y_pred_classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_orig, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    
    return test_acc, y_pred_classes

# Hyperparameter tuning (simplified)
def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape, num_classes):
    print("\n=== Hyperparameter Tuning for CNN ===")
    
    # Define hyperparameter combinations to try
    learning_rates = [0.001, 0.0005]
    dropout_rates = [0.25, 0.5]
    batch_sizes = [32, 64]
    
    best_val_acc = 0
    best_params = {}
    
    for lr in learning_rates:
        for dropout in dropout_rates:
            for batch_size in batch_sizes:
                print(f"\nTrying: lr={lr}, dropout={dropout}, batch_size={batch_size}")
                
                try:
                    # Build a smaller CNN for quick tuning
                    model = Sequential([
                        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(dropout),
                        
                        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(dropout),
                        
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(dropout),
                        Dense(num_classes, activation='softmax')
                    ])
                    
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Train for fewer epochs during tuning
                    history = model.fit(
                        X_train, y_train,
                        epochs=10,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        verbose=0
                    )
                    
                    # Get the validation accuracy
                    val_acc = max(history.history['val_accuracy'])
                    print(f"Validation accuracy: {val_acc:.4f}")
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {
                            'learning_rate': lr,
                            'dropout_rate': dropout,
                            'batch_size': batch_size
                        }
                except Exception as e:
                    print(f"Error during hyperparameter combination: {e}")
                    continue
    
    # Return default parameters if no successful runs
    if not best_params:
        print("\nAll hyperparameter combinations failed. Using default parameters.")
        best_params = {
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'batch_size': 32
        }
    else:
        print("\nBest parameters:", best_params)
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return best_params

def main():
    # Load data
    print("Loading data...")
    X_train = np.load('Assignment2Data/X_train.npy')
    X_test = np.load('Assignment2Data/X_test.npy')
    y_train = np.load('Assignment2Data/y_train.npy')
    y_test = np.load('Assignment2Data/y_test.npy')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Data exploration
    num_classes = explore_data(X_train, y_train)
    
    # Preprocess data
    X_train_proc, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train_orig, y_val_orig, y_test_orig = preprocess_data(
        X_train, X_test, y_train, y_test, num_classes
    )
    
    # Get input shape
    input_shape = X_train_proc.shape[1:]
    
    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train_proc, y_train_cat, X_val, y_val_cat, input_shape, num_classes)
    
    # Train MLP model
    mlp_model = build_mlp(input_shape, num_classes)
    mlp_acc, mlp_pred = train_and_evaluate(
        mlp_model, "MLP", X_train_proc, X_val, y_train_cat, y_val_cat, X_test, y_test_cat, y_test_orig
    )
    
    # Train CNN model with best parameters
    cnn_model = build_cnn(input_shape, num_classes)
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    cnn_acc, cnn_pred = train_and_evaluate(
        cnn_model, "CNN", X_train_proc, X_val, y_train_cat, y_val_cat, X_test, y_test_cat, y_test_orig
    )
    
    # Compare models
    print("\n=== Model Comparison ===")
    print(f"MLP Test Accuracy: {mlp_acc:.4f}")
    print(f"CNN Test Accuracy: {cnn_acc:.4f}")
    
    # Analyze examples where models disagree
    disagree_idx = np.where(mlp_pred != cnn_pred)[0]
    print(f"Number of examples where models disagree: {len(disagree_idx)}")
    
    if len(disagree_idx) > 0:
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(disagree_idx[:5]):  # Show first 5 disagreements
            plt.subplot(1, 5, i+1)
            plt.imshow(X_test[idx])
            plt.title(f"True: {y_test_orig[idx]}\nMLP: {mlp_pred[idx]}\nCNN: {cnn_pred[idx]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('model_disagreements.png')

if __name__ == "__main__":
    main() 