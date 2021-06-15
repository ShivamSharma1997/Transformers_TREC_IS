import re
import sys
import json
import math
import string
import gzip

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile

from scipy.special import softmax

from simpletransformers.classification import MultiLabelClassificationModel

####################### DEFINING PARAMETERS #######################

model_path = './models/RoBERTa10/'
file_path = './data/test/2020B_main'
cat_dict_path = './data/TASK1/TASK1_categorical.json'

####################### MAIN CODE #######################

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def clean_tweet(text):

    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'

    text = re.sub(giant_url_regex, 'URLHERE', text)
    text = re.sub(mention_regex, 'MENTIONHERE', text)

    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    text = re.sub('RT', '', text)
    return text

with open(cat_dict_path) as fp:
    category_to_label = json.load(fp)

label_to_category = {}

label_to_category[0] = 'Tweet ID'

for cat in category_to_label.keys():
    label_to_category[category_to_label[cat]+1] = cat


argument_dict = {'reprocess_input_data': True,
                 'overwrite_output_dir': True,
                 'num_train_epochs': 10}

model = MultiLabelClassificationModel('roberta',
                                     model_path,
                                     args=argument_dict)

id_list, text_list = [], []

with open(file_path) as fp:

  while True:
    data = fp.readline()

    if len(data) == 0:
      break

    data = json.loads(data)
    text = clean_tweet(data['allProperties']['text'])
    tweet_id = data['allProperties']['id']

    text_list.append(text)
    id_list.append(tweet_id)

_, raw_outputs = model.predict(text_list)

predictions = []

for output in raw_outputs:
    temp = []
    for val in output:
        temp.append(val)
    predictions.append(temp)

pred_list = []

for tweet_id, weights in zip(id_list, predictions):
    temp = [tweet_id]
    temp.extend(weights)
    pred_list.append(temp)

result_df = pd.DataFrame(pred_list)
resut_df = result_df.rename(columns=label_to_category)
result_df.to_csv('./outputs/SUB1/TASK1/information_type.csv', index=False)
