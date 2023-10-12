TPV_PATH_DATASET=$1
TPV_PATH_MNIST_DATASET=$2

ln -s $TPV_PATH_DATASET files
ln -s $TPV_PATH_MNIST_DATASET ./notebooks/other_datasets/mnist_brain_digits/files
