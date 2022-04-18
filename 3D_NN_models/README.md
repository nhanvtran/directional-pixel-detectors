# Analog Version - adding time info

- We have time info. for 16 time slices (think of as them as video frames). We think the angular direction information mostly comes from the first 8 frames, so will make a prediction based on the first eight frames.
- Inputs to the model: 8 frames per sample
- Output: predict cotBeta
- Model: current model is a CNN with time-distributed layers, followed by LSTM

## How to run the code
- Run datagen.py for training or testing (decide on train/test split in advance, assuming all clusters were randomly generated)
- Create cluster train/test sets in two different directories. Each cluster has its own own folder (cluster0, cluster1, etc). Each cluster folder has 8 frames using .npy format
- Run prepareFileList.ipynb to create a list of samples (will have one for training, one for test). Each sample has 8 frames in it. The list of samples will be used for batched input. The NN will grab samples from the sample list, instead of loading all samples into memory at once.
- Run CNN-LSTM-april18.ipynb 
- Note: need to change paths for each folder for your system
