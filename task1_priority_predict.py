import re
import sys
import json
import string
import gzip

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile

from simpletransformers.classification import ClassificationModel

####################### DEFINING PARAMETERS #######################

model_path = './models/BERT10/'
file_path = './data/test/2020B_main'

####################### MAIN CODE #######################

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


args = {'fp16':False,
        'show_running_loss' : True,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'regression': True,
        'num_train_epochs': 10}

model = ClassificationModel('bert',
                             model_path,
                             num_labels=1,
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

predictions, _ = model.predict(text_list)

with open('./outputs/SUB1/TASK1/priority.csv', 'w') as fp:
    fp.write("Tweet ID, Priority Score")
    for tweet_id, pred in zip(id_list, predictions):
        fp.write('{},{}\n'.format(tweet_id, pred))
