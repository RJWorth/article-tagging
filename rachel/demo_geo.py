import os
import sys

### Move to directory containing this file
# print(os.path)
# print(os.path.split(__file__)[0])
os.chdir('/home/rachel/Code/civic_data/article-tagging/rachel/')

import glob
saved_files = glob.glob('saved/weights*.hdf5')
if saved_files:
    delete = input(('This will delete existing saved weight'
                    ' files, proceed? [y/n] '))
    while delete not in ['y', 'n']:
        delete = input(('This will delete existing saved weight'
                        ' files, proceed? [y/n] '))
    if delete == 'y':
        for f in saved_files:
            os.remove(f)
    else:
        print('Exiting.')
        exit()

from lib.tagnews import utils
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import json
import requests
import keras
pd.set_option('display.width', 180)
pd.set_option('max.columns', 15)

### Add len=3 for running interactively
if len(sys.argv) in [1, 3]:
    num_epochs = 20
else:
    num_epochs = int(sys.argv[1])

### Get pre-trained word vectorizer (400,000 rows/words, 50 columns
glove = utils.load_vectorizer.load_glove('../lib/tagnews/data/glove.6B.50d.txt')
# ner = utils.load_data.load_ner_data('../../../data/')  # ???

### 1,348,863-character string with 0\n between each word
with open('../lib/tagnews/geoloc/models/lstm/training.txt', encoding='utf-8') as f:
    training_data = f.read()
### shape = (183206, 2), each word is on its own line and the number after it is the tag (0 or 1)
training_df = pd.DataFrame([x.split() for x in training_data.split('\n') if x],
                           columns=['word', 'tag'])
training_df.iloc[:, 1] = training_df.iloc[:, 1].apply(int)
training_df['all_tags'] = 'NA'

### Add columns with whether each word starts uppercase (0), and vectorization of word (0-49, so two 0s)
ner = training_df # pd.concat([training_df, ner]).reset_index(drop=True)
ner = ner[['word', 'all_tags', 'tag']]
ner = pd.concat([ner,
                 pd.DataFrame(ner['word'].str[0].str.isupper().values),
                 pd.DataFrame(glove.reindex(ner['word'].str.lower()).values)],
                axis='columns')
ner.fillna(value=0.0, inplace=True)

### Settings for modelling?
data_dim = 51
timesteps = 25 # only during training, testing can take arbitrary length.
num_classes = 2

### 19/20 Train/test split, and also break data up into 25 chunks
### 3D array, (new number of rows, timesteps, data_dim)
train_val_split = int(19 * ner.shape[0] / 20.)
### count up by 25 to just below the split
ner_train_idxs = range(0, train_val_split - timesteps, timesteps)
x_train = np.array([ner.iloc[i:i+timesteps, 3:].values
                    for i in ner_train_idxs])
y_train = np.array([to_categorical(ner.iloc[i:i+timesteps, 2].values, 2)
                    for i in ner_train_idxs])

ner_val_idxs = range(train_val_split, ner.shape[0] - timesteps, timesteps)
x_val = np.array([ner.iloc[i:i+timesteps, 3:].values
                  for i in ner_val_idxs])
y_val = np.array([to_categorical(ner.iloc[i:i+timesteps, 2].values, 2)
                  for i in ner_val_idxs])
