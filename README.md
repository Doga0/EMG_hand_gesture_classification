# EMG Hand Gesture Classification

## Project Overview
This project aims to classify hand gestures using Electromyography (EMG) signals. The system collects EMG data from muscle activity, processes the signals, and uses machine learning to recognize specific hand gestures. The implementation includes data collection, preprocessing, and model training components.

## Repository Structure
    EMG_hand_gesture_classification/
    │
    ├── dataCollector.py          	# Script for collecting EMG data
    ├── preprocess.ipynb          	# Jupyter notebook for data preprocessing
    ├── wav_to_freq.ipynb         	# Jupyter notebook for frequency domain conversion
    ├── model.py                  	# Deep learning model implementation
    └── README.md                 	# Project documentation

## Components

### 1. Data Collection (`dataCollector.py`)

![Data Collector](https://github.com/Doga0/EMG_hand_gesture_classification/blob/main/dataCollector.png)

This script handles the collection of EMG data from sensors. A helper tool to automatically collect and label signal data in real time.
**Features:**

-   Records EMG signals in real-time
-   Saves data in CSV format for processing
-   Configurable recording duration and channels

### 2. Data Preprocessing (`preprocess.ipynb`)

Jupyter notebook preparing EMG data for analysis.

### 3. Frequency Domain Conversion (wav_to_freq.ipynb)
Converts time-domain EMG signals to frequency-domain representations.

### 4. Machine Learning Model (`model.py`)

Implementation of the gesture classification model.

**Features:**

-   LSTM architecture for time-series classification
-   Configurable hyperparameters
-   Training and evaluation   
