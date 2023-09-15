# total-perspective-vortex

# Description

This project aims to classify states from EEG data acquired from PhysioNet. The primary goal is to design an accurate classification system using decomposition algorithms and machine learning techniques.

## Decomposition algorithm

In this project, we utilize various decomposition algorithms to preprocess the EEG data and extract features for classification. Two prominent algorithms are employed:

### CSP (Common Spatial Patterns)

Common Spatial Patterns (CSP) algorithm is employed to extract discriminative spatial patterns from EEG data, enhancing the classification of different states.

### ICA (Independent Component Analysis)

Independent Component Analysis (ICA) is used to separate EEG signals into statistically independent components, aiding in the identification of distinct patterns related to different states.

# Usage

To utilize this project, follow these steps:

1. Install the required dependencies mentioned in the project documentation.
2. Execute the preprocessing, decomposition, and training components as per the provided instructions.

# Results

## Mandatory part

The mandatory aspect of the project involves utilizing the decomposition algorithms to classify states. The following table summarizes the results:

With filter 
CSP + ICA:

| Model                              | Accuracy |
|------------------------------------|---------|
| Linear discriminant analysis (LDA)  | 0.86    |
| Decision tree                       | 0.76    |
| K-nearest neighbors (KNN)           | 0.70    |



## Bonus part

In the bonus part, we explored additional methods involving machine learning and deep learning techniques.

### Deep learning

Deep learning models, including convolutional neural networks (CNNs) and recurrent neural networks (RNNs), were trained to further enhance classification accuracy. The following table summarizes the deep learning results:

| Model | Accuracy |
|-------|----------|
| GCN   | 0.2549   |
| CNN   | 0.2392   |
| LSTM  | 0.2392   |

### Machine learning

We trained various machine learning models, such as SVM and Random Forest, on preprocessed EEG data. The following table summarizes the machine learning results:

We have use the following features for the machine learning models:

- Average
- Root Mean Square
- Standard Deviation
- Variance
- Entropy
- Energy
- Discrete Wavelet Transform
- Power Spectral Density

| Model                              | Accuracy |
|------------------------------------|----------|
| Linear discriminant analysis (LDA) | 0.57     |
| SVM                                | 0.47     |
| KNN                                | 0.38     |


With mne-feature we have use the following features for the machine learning models:

- mean
- variance
- std
- spect_edge_freq
- kurtosis
- zero_crossings
- spect_slope

| Model                              | Accuracy |
|------------------------------------|----------|
| Linear discriminant analysis (LDA) | 0.50     |
| SVM                                | 0.42     |
| KNN                                | 0.42     |


Based on this [paper](https://arxiv.org/ftp/arxiv/papers/1312/1312.2877.pdf) we have extract the following features:

3 epochs:
- from -2 to 0 with filter from 8 to 30
- from 4.1 to 5.1 with filter from 8 to 30
- from -2 to 0 with filter all to 3

After that we have extract the following features from the ICA of each 3 epochs:

- mean
- std
- power
- energy
- variance
- root_mean_square

| Model         | Accuracy |
|---------------|----------|
| Decision Tree | 0.81     |
| MLP           | 0.54     |



# Working with other dataset

### MNIST Brain Digits

We have try to train with the MNIST Brain Digits dataset. We have try to train to predict the exact digit and to predict the digit class (0-9).

| Model         | Accuracy |
|---------------|----------|
| Random Forest | 0.25     |
| SVM           | 0.23     |
| MLP           | 0.23     |
| Decision Tree | 0.20     |
| KKN           | 0.19     |


If digits is digits or even or odd:

| Model             | Accuracy |
|-------------------|----------|
| Random Forest     | 0.51     |
| XGB               | 0.51     |
| MLP               | 0.50     |
| Gradient Boosting | 0.49     |
| KKN               | 0.48     |
| Decision Tree     | 0.45     |
| LDA               | 0.38     |

With the bad results we have try to predict if the digit is a digit or not.

| Model             | Accuracy |
|-------------------|----------|
| Random Forest     | 0.87     |
| Gradient Boosting | 0.85     |
| XGB               | 0.85     |
| SVM               | 0.75     |
| Decision Tree     | 0.80     |
| KKN               | 0.82     |
| LDA               | 0.76     |
| MLP               | 0.70     |


# Annexes

[Sources and papers](papers.md)