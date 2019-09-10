
### Copy of lib/crimetype/models/binary_stemmed_logistic/save_model.py
### with my own modifications and comments as I figure out what it's doing

import os
import time
import sys

import numpy as np
import sklearn
# import sklearn.feature_extraction.text
import sklearn.multiclass
import sklearn.linear_model
from sklearn.model_selection import train_test_split

from lib.tagnews.utils import load_data as ld
from lib.tagnews.utils.model_helpers import LemmaTokenizer

# needed to make pickle-ing work
from nltk import word_tokenize  # noqa
from nltk.stem import WordNetLemmatizer  # noqa

np.random.seed(1029384756)

### If an argument was provided, this uses that as the number of rows, otherwise probably loads all?
### Running in command line, this doens't work. Added new option to use the default line
if len(sys.argv) == 2:
    df = ld.load_data(nrows=int(sys.argv[1]))
elif len(sys.argv) == 1:
    df = ld.load_data()
elif len(sys.argv) == 3:
    ### used when running from interactive shell
    df = ld.load_data()
else:
    raise Exception('BAD ARGUMENTS')

### Get subset of columns which are 0 or 1, if any is 1 in a row, use that row
### In small dataset, all but one are used, ~200. In larger dataset, ~56k rows
### (Converted .ix to .loc)
crime_df = df.loc[df.loc[:, 'OEMC':'TASR'].any(1), :]
### Now append a sample of all the "relevant" rows (even if they're duplicates? why?)
crime_df = crime_df.append(
    df.loc[~df['relevant'], :].sample(n=min(3000, (~df['relevant']).sum()),
                                      axis=0)
)

### Create vectorizer object (binary => either 0 or 1 for each word, instead of a count)
vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=LemmaTokenizer(),
    binary=True,
    max_features=40000
)
### Create classifier object
clf = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression(verbose=0)
)

### Convert article text into sparse matrix. input shape=(212,), output .data shape = (52558,)?
X = vectorizer.fit_transform(crime_df['bodytext'].values)
### 38 columns, presumably the types of articles.
y = crime_df.loc[:, 'OEMC':'TASR'].values

### Add train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

### Fit the model, obviously
clf.fit(X_train, y_train)

### Get class that manages the modeling of the article tags (Why, when it's never used again? just for demo?)
from lib.tagnews.crimetype.tag import CrimeTags
crimetags = CrimeTags(clf=clf, vectorizer=vectorizer)
### demonstration of getting the tag probabilities for a sample text
print(crimetags.tagtext_proba(('This is an article about drugs and gangs.')))

### Get test score
pred_test = clf.predict(X_test)
compare_test = (pred_test == y_test)
print('Fraction of tags that agree: {:.2%}'.format(compare_test.mean()))
print('Fraction of true tags that are predicted correctly: {:.2%}'.format(((pred_test == y_test) & (y_test == 1)).sum()/(y_test == 1).sum()))
print('Fraction of false tags that are predicted correctly: {:.2%}'.format(((pred_test == y_test) & (y_test == 0)).sum()/(y_test == 0).sum()))
print('True tags: {};  predicted true tags: {}'.format(y_test.sum(), pred_test.sum()))

### Save model and vector objects as pickle files
import pickle
curr_time = time.strftime("%Y%m%d-%H%M%S")
classifier_file = os.path.join('rachel', 'crimetags', os.path.split(__file__)[0], 'model-' + curr_time + '.pkl')
vectorizer_file = os.path.join('rachel', 'crimetags', os.path.split(__file__)[0], 'vectorizer-' + curr_time + '.pkl')
print('Saving model and vectorizer:\n  {}\n  {}'.format(classifier_file, vectorizer_file))
with open(classifier_file, 'wb') as f:
    pickle.dump(clf, f)
with open(vectorizer_file, 'wb') as f:
    pickle.dump(vectorizer, f)

t_ed = time.time()
print('Runtime: {} m'.format((t_ed-t_st)/60.))