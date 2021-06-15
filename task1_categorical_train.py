import logging
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

from simpletransformers.classification import MultiLabelClassificationModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_pickle('./data/TASK1/Information_Type/train_aug_df.pkl')
val_df = pd.read_pickle('./data/TASK1/Information_Type/val_aug_df.pkl')

print(train_df.head())

output_dir = './models/RoBERTa10/'

args = {'fp16':False,
        'show_running_loss' : True,
        'output_dir' : output_dir,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 10}

model = MultiLabelClassificationModel('roberta', 'roberta-base',
                                      num_labels=25,
                                      args=args)

model.train_model(train_df)

preds,_ = model.predict(val_df['Text'])

preds = list(map(np.round, preds))
y_true = val_df['Labels'].values

preds = np.array(list(map(list, preds)))
y_true = np.array(list(map(list, y_true)))

acc = accuracy_score(y_true, preds)
f_score = f1_score(y_true, preds, average='macro')

print('Accuracy:', acc)
print('F1 Score:', f_score)
