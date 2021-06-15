import logging
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

from simpletransformers.classification import ClassificationModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_pickle('./data/TASK1/Priority_Type/train_aug_df.pkl')
val_df = pd.read_pickle('./data/TASK1/Priority_Type/val_aug_df.pkl')

print(train_df.head())

output_dir = './models/BERT10/'

args = {'fp16':False,
        'show_running_loss' : True,
        'output_dir' : output_dir,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'regression': True,
        'num_train_epochs': 10}

model = ClassificationModel('bert', 'bert-base-uncased',
                            num_labels=1,
                            args=args)

model.train_model(train_df)

preds,_ = model.predict(val_df['Tweet ID'])

y_true = val_df['Labels'].values

acc = accuracy_score(y_true, preds)
f_score = f1_score(y_true, preds, average='macro')

print('\nAccuracy:', acc)
print('F1 Score:', f_score)
