TPV_PATH_PHYSIONET_MOV_DATASET=$1
TPV_PATH_MNIST_DATASET=$2
TPV_PATH_SLEEP_PHYSIONET_DATASET=$3

ln -s $TPV_PATH_PHYSIONET_MOV_DATASET files
ln -s $TPV_PATH_MNIST_DATASET ./notebooks/other_datasets/mnist_brain_digits/files
ln -s $TPV_PATH_SLEEP_PHYSIONET_DATASET ./notebooks/other_datasets/sleep_edf/files
