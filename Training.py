import random
import json
import pickle
import numpy as np

import nltk # Natural language Toolkit
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
# It lemitizes the words like (work=working=worked=works)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lematizer = WordNetLemmatizer()
intents = json.loads(open("script.json").read())

# doing some filtering of the words

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

# separating the words with their

for intent in intents["intents"]:
    for pattern in intent['patterns']:
        word_list = wordpunct_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lematizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

print(words)