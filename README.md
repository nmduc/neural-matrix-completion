# neural-matrix-completion

This repository contains the source code for the paper entitled "Extendable Neural Matrix Completion" (published at ICASSP 2018)

## Prerequisites
```
- Tensorflow 
- Numpy 
```
The code has been tested in Ubuntu 18.04 and MacOSX, with
```
- Python 3.5
- Tensorflow v.1.8
- Numpy v.1.14 
```
You should be able to run the code with other versions of Tensorflow and Numpy as well. 

## Usage
### Prepare data
A sample split (75% training, 5% validation, 20% testing) of the MovieLens100K dataset is included (inside folder ./data/MovieLens100K/):
- rating.npz: original matrix 
- train_mask.npz: training mask, same shape as the original matrix
- val_mask.npz: validation mask, same shape as the original matrix
- test_mask.npz: testing mask, same shape as the original matrix

All these matrices are saved as Numpy sparse csr_matrix. 

Alternatively, you can write your own DataLoader to load and feed the model during training, with data prepared in your preferred way.

### Configurations
The configs/configs_ML100K.py file sets all the default configurations and hyper-parameters.
There are some other hyperparameters inside train.py and test.py to set the input and output directories.
To use your own configurations, you can either edit these files, or override the default configurations using flags.

For example:
```
python train.py --output_basedir ./outputs/MovieLens100K/
```
### Training
After preparing your dataset and set all the necessary hyper-parameters, you are ready to train your model.
Run train.py, together with our hyper-parameters, to start the training. For example:
```
python train.py --data_dir=./data/MovieLens100K/ --output_basedir=./outputs/
```
The trained model will be saved into outputs/snapshots.

### Testing
After training the model, you can test it by calling test.py, with the default parameters or with your own parameters. For example:
```
python test.py --data_dir=./data/MovieLens100K/ --snapshot_dir=./outputs/snapshots/
```
## Related
If you want to produce discrete output, you may want to check our paper "Learning Discrete Matrix Factorization Models" (published at IEEE Signal Processing Letters 2018). 
The source code is available at: https://github.com/nmduc/discrete-matrix-factorization

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Reference
If you find the source code useful, please cite us:
```
D. M. Nguyen, E. Tsiligianni and N. Deligiannis, "Extendable Neural Matrix Completion," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 6328-6332.
```
