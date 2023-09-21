# total-perspective-vortex

# Description

This project aims to classify states from EEG data acquired from PhysioNet. The primary goal is to design an accurate classification system using decomposition algorithms and machine learning techniques.

## Decomposition algorithm

In this project, we utilize various decomposition algorithms to preprocess the EEG data and extract features for classification. Two prominent algorithms are employed:

### CSP (Common Spatial Patterns)

Common Spatial Patterns (CSP) algorithm is employed to extract discriminative spatial patterns from EEG data, enhancing the classification of different states.

The goal of the CSP algorithm is to find spatial filters that maximize the variance of one class of signals while minimizing the variance of another class. This is achieved by finding a transformation matrix that, when applied to the raw EEG signals, results in new signals (features) that have the highest variance for one class and the lowest variance for the other.

### ICA (Independent Component Analysis)

Independent Component Analysis (ICA) is a statistical and computational technique used to separate a multivariate signal into its constituent independent subcomponents. It's particularly useful when the observed signals are a linear mixture of various independent sources, making it a powerful tool in signal processing, neuroscience, image processing, and other fields.


# Usage and notebooks

## Notebooks

Notebooks are located in the `/notebooks` folder. They are used to preprocess the data and train the models and lot of analysis. There are following directories:

```
/notebooks <-- Notebooks used to present the mandatory part and bonus part
  /decomposition_alg <-- Notebooks used to present the decomposition algorithms
  /research <-- Notebooks used to present the research to improve the results of the mandatory part, there is also the deep learning part
  /other_dataset <-- Notebooks used to present the results with MNIST Brain Digits dataset
```

## Scripts process

The scripts are located in the `/process` folder. They are used to preprocess the data and train the models.
There are following directories:

```
/process
  /dl <-- In this directory we can find the scripts used to train and predict the deep learning models
  /ml <-- In this directory we can find the scripts used to train and predict the machine learning models
```

There is usage of the scripts:

### Machine learning

**Train script:**
```
python train.py -h
usage: train.py [-h] -s SUBJECT [-e {hands_vs_feet,left_vs_right,imagery_left_vs_right,imagery_hands_vs_feet}]
                [-d DIRECTORY_DATASET] [-m MODEL] [-o OUTPUT] [-da DECOMPOSITION_ALGORITHM] [-nsmdl] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -s SUBJECT, --subject SUBJECT
                        Subject number, sequence of subjects (separated by comma) or all
  -e {hands_vs_feet,left_vs_right,imagery_left_vs_right,imagery_hands_vs_feet}, --experiment {hands_vs_feet,left_vs_right,imagery_left_vs_right,imagery_hands_vs_feet}
                        Type training
  -d DIRECTORY_DATASET, --directory-dataset DIRECTORY_DATASET
                        Directory dataset
  -m MODEL, --model MODEL
                        Model name.
                        Availables models: gradient_boosting,lda,svc,knn,random_forest,mlp,decision_tree,xgb
  -o OUTPUT, --output OUTPUT
                        Output path file
  -da DECOMPOSITION_ALGORITHM, --decomposition-algorithm DECOMPOSITION_ALGORITHM
                        Decomposition algorithm.
                        Available: TurboCSP,MNECSP
  -nsmdl, --no-save-model
                        Save model
  -v, --verbose         Verbose                                                                                          ~5s 
```

**Prediction script:**
```
python predict.py -h
usage: predict.py [-h] [-strm] [-e {all,hands_vs_feet,left_vs_right,hands_vs_feet,left_vs_right}] [-o OUTPUT_FILE] -s
                  SUBJECT [-m MODEL_PATH] [-d DIRECTORY_DATASET] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -strm, --stream-mode  Stream mode, When this flag is enabled, the program will wait for the data from the server at port 5000.
  -e {all,hands_vs_feet,left_vs_right,hands_vs_feet,left_vs_right}, --experiment {all,hands_vs_feet,left_vs_right,hands_vs_feet,left_vs_right}
                        Type training
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file
  -s SUBJECT, --subject SUBJECT
                        Subject number, sequence of subjects (separated by comma) or all
  -m MODEL_PATH, --model-path MODEL_PATH
                        Model path
  -d DIRECTORY_DATASET, --directory-dataset DIRECTORY_DATASET
                        Directory dataset
  -v, --verbose         Verbose 
```

You can also use ml prediction with stream server. The stream server is located in the `/process/ml/stream/main.py` file. There is an example of usage with prediction script:

**Stream server:**

```bash-session
cd process/ml/stream
python main.py -d ../../files
```

**Run prediction script:**
```bash-session
cd process/ml
python predict.py -strm -s 1,2
```

### Deep learning

**Train script:**
```
python train.py -h
usage: train.py [-h] -ns NUMS_SUBJECTS [-t {all,hands_vs_feet,left_vs_right,imagery_hands_vs_feet,imagery_left_vs_right}]
                -r RATIO [-e EPOCHS] [-bs BATCH_SIZE] [-d DIRECTORY_DATASET] [-m MODEL] [-o OUTPUT] [-nsmdl] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -ns NUMS_SUBJECTS, --nums-subjects NUMS_SUBJECTS
                        Numbers of subjects
  -t {all,hands_vs_feet,left_vs_right,imagery_hands_vs_feet,imagery_left_vs_right}, --type-training {all,hands_vs_feet,left_vs_right,imagery_hands_vs_feet,imagery_left_vs_right}
                        Type training
  -r RATIO, --ratio RATIO
                        Ratio of train/test
  -e EPOCHS, --epochs EPOCHS
                        Epochs
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -d DIRECTORY_DATASET, --directory-dataset DIRECTORY_DATASET
                        Directory dataset
  -m MODEL, --model MODEL
                        Model name.
                        Availables models: cnn2d_classic, gcn_classic, cnn2d_advanced
  -o OUTPUT, --output OUTPUT
                        Output path file
  -nsmdl, --no-save-model
                        Save model
  -v, --verbose         Verbose 
```

**Prediction script:**

```
python predict.py -h
usage: predict.py [-h] [-t {all,hands_vs_feet,left_vs_right,hands_vs_feet,left_vs_right}] [-o OUTPUT_FILE] -s SUBJECT
                  [-m MODEL_PATH] [-d DIRECTORY_DATASET] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -t {all,hands_vs_feet,left_vs_right,hands_vs_feet,left_vs_right}, --type-training {all,hands_vs_feet,left_vs_right,hands_vs_feet,left_vs_right}
                        Type training
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file
  -s SUBJECT, --subject SUBJECT
                        Subject number, sequence of subjects (separated by comma) or all
  -m MODEL_PATH, --model-path MODEL_PATH
                        Model path
  -d DIRECTORY_DATASET, --directory-dataset DIRECTORY_DATASET
                        Directory dataset
  -v, --verbose         Verbose
```

# Results

## Mandatory part

The mandatory aspect of the project involves utilizing the decomposition algorithms to classify states.

The preprocessing steps are as follows:

- Apply a bandpass filter to the raw EEG data
- Apply a notch filter to remove powerline noise
- Apply a decomposition algorithm ICA to extract independent components
- Add pipeline with CSP -> <model> to classify the states

The following table summarizes the results of the three best models:

| Model                              | Accuracy |
|------------------------------------|---------|
| Linear discriminant analysis (LDA)  | 0.86    |
| Decision tree                       | 0.76    |
| K-nearest neighbors (KNN)           | 0.70    |


## Bonus part

In the bonus part, I explored additional methods involving machine learning and deep learning techniques and feature extraction methods to further enhance the classification accuracy.

### Deep learning

Deep learning models, including convolutional neural networks (CNNs) and recurrent neural networks (RNNs) like long short-term memory (LSTM), were trained on preprocessed EEG data. The following table summarizes the deep learning results:

| Model | Accuracy |
|-------|----------|
| GCN   | 0.2549   |
| CNN   | 0.2392   |
| LSTM  | 0.2392   |

There is not a really good results with the deep learning models. We have try to train with the MNIST Brain Digits dataset and we have better results. You can see the results in the section "Working with other dataset".

### Machine learning

We trained various machine learning models, such as SVM and Random Forest, on preprocessed EEG data.

We have use the following features for the machine learning models:

- Average
- Root Mean Square
- Standard Deviation
- Variance
- Entropy
- Energy
- Discrete Wavelet Transform
- Power Spectral Density


The following table summarizes the machine learning results:

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

The following table summarizes the machine learning results:

| Model                              | Accuracy |
|------------------------------------|----------|
| Linear discriminant analysis (LDA) | 0.50     |
| SVM                                | 0.42     |
| KNN                                | 0.42     |


Based on this [paper](https://arxiv.org/ftp/arxiv/papers/1312/1312.2877.pdf) I have try to improve the results of the classification of the mandatory part.

The principle is to extract 3 types of epochs:

- **ERD** from -2 to 0 
- **ERS** from 4.1 to 5.1
- **MRCP** from -2 to 0

For ERD and ERS I used a filter to extract the frequency between 8 and 30 Hz. For MRCP I used a filter to extract the frequency under 3 Hz.

After that for each epochs I have apply the ICA algorithm to extract the independent components.
From this independent components I have able to calculate the `activation_vector`. From this vector I have extract the following features:

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

I have try to train with the MNIST Brain Digits dataset. I have try to predict the digit class (0-9).

| Model         | Accuracy |
|---------------|----------|
| Random Forest | 0.25     |
| SVM           | 0.23     |
| MLP           | 0.23     |
| Decision Tree | 0.20     |
| KKN           | 0.19     |


When I see the bad results I have try to predict if the digit is even, odd or a digit.

| Model             | Accuracy |
|-------------------|----------|
| Random Forest     | 0.51     |
| XGB               | 0.51     |
| MLP               | 0.50     |
| Gradient Boosting | 0.49     |
| KKN               | 0.48     |
| Decision Tree     | 0.45     |
| LDA               | 0.38     |

And after that I have try to predict if the digit is a digit or not.

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


I have also try to used the deep learning for this dataset but I have not good results.

# Annexes

[Sources and papers](papers.md)