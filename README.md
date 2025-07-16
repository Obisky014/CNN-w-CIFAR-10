# CNN Model Development for CIFAR-10 

# Overview

This little project is my attempt to build a CNN using TensorFlow/Keras to classify images from the CIFAR-10 dataset. CIFAR-10 has 60,000 color images (32x32 pixels) across 10 classes like airplane, dog, frog, truck, etc. I started with a basic CNN setup and gradually tried to make it better—mainly by battling overfitting and making the model generalize more to unseen data.

# 1. Initial Model Setup

Dataset: Loaded using tf.keras.datasets.cifar10.load_data().

**Preprocessing:**

Images normalized to [0, 1]

Labels one-hot encoded using tf.keras.utils.to_categorical, so that I don't use sparse_categorical_crossentropy as the loss function when compiling the model

**Model Architecture:**
A simple Sequential CNN:

3 × Conv2D layers (32 → 64 → 64 filters), all ReLU

2 × MaxPooling2D

Flatten → Dense(128) + ReLU → Dropout(0.5) → Dense(10) + softmax

**Training:**
Compiled with Adam optimizer and categorical_crossentropy, trained for 20 epochs with a validation split.

# 2. Spotting and Tackling Overfitting
I noticed early signs of overfitting, albeit a MILD one:

**Training accuracy:** 0.70

**Validation accuracy:** 0.65

So I made a bunch of changes:

**Image Augmentation:** Used ImageDataGenerator with rotations, shifts, zooms, and flips to make training data more varied.

**Dropout:** Increased from 0.5 to 0.6 to regularize harder.

**BatchNormalization:** Added after each Conv2D layer to stabilize and speed up training.

**EarlyStopping:** Used EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True) to stop training once validation performance stopped improving.

With these changes, I bumped training epochs to 50, but training usually stopped earlier (like between epochs 5, 6 and 13...this was also a result of me restarting the kernel). Final test accuracy: ~72%. Overfitting gap significantly reduced.

# 3. Performance Analysis (Confusion Matrix Stuff)
I compared two confusion matrices—one before and one after my overfitting fixes.

**Before:**

Great at classifying truck (871) and automobile (860)

Struggled with bird (517) — lots of confusion with similar animal classes

**After:**

Better generalization overall:

airplane up to 813

cat and bird slightly improved

truck performance shot up to 938

dog performance dropped (764 → 567), maybe I went a bit too hard on regularization?


# Requirements
Python 3.x

TensorFlow Library

NumPy

Matplotlib

Seaborn (for plotting confusion matrices)

How to Use This Project
Clone the repo or download the notebook

Run:
pip install tensorflow numpy matplotlib seaborn

Load CIFAR-10 and run the initial model

Try the overfitting fixes and retrain

(Optional) Add your own 32x32 image files and test predictions

Future Plans
Improve dog classification (maybe by tuning dropout or adding more layers)

Try transfer learning with pretrained models like ResNet

Play around with CIFAR-100 (100 classes instead of 10)

# Acknowledgments
This was mostly me, trying to improve my knowledge on data science and machine learning. That's it really
This wasn't built from some perfect plan—just me experimenting, breaking stuff, Googling errors, and learning along the way. Hopefully, it helps someone else doing the same.
