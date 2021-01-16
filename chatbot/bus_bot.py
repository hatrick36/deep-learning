import json
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import nltk.stem
import tflearn
import random
import pickle
import os

nltk.download('punkt')

print('Number of GPUs available:', len(tf.config.experimental.list_physical_devices('GPU')))

# load in json data file
with open('data.json') as file:
    data = json.load(file)

stemmer = LancasterStemmer()
# after first succesful run of program lists will be loaded from pickle file
try:
    with open('data.pickle', 'rb') as f:
        phrases, labels, training, output = pickle.load(f)
# first run through
except:
    phrases = []
    labels = []
    docs_x = []
    docs_y = []
    # tolkenizes patterns in json file appends words to docs_x and tags to docs_y
    for x in data['data']:
        for pattern in x['patterns']:
            words = nltk.word_tokenize(pattern)
            phrases.extend(words)
            docs_x.append(words)
            docs_y.append(x['tag'])
        if x['tag'] not in labels:
            labels.append(x['tag'])
    # optional code uncomment if model not preforming for insight
    # print('phrases:')
    # print(phrases)
    # print('labels:')
    # print(labels)
    #print('docs_x:')
    #print(docs_x)
    # print('docs_y')
    # print(docs_y)

    # stems words in phrases and puts to lowercase
    phrases = [stemmer.stem(w.lower()) for w in phrases if w != '?']
    # appends set of phrases to training list with output being corresponding labels for phrases
    phrases = sorted(list(set(phrases)))
    labels = sorted(labels)
    training = []
    output = []

    # creates vector of zeros and ones to determine tag
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        # stem all words in patterns
        words = [stemmer.stem(w) for w in doc]
        for w in phrases:
            # create vector of existent words vs non-existent
            if w in words:
                bag.append(1)
            else:
                bag.append(0)
            # set output_row to list of out empty
            output_row = out_empty[:]
            # find index of tag in labels and set value to 1 in output_row
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)
    training = np.array(training)
    output = np.array(output)
    # send lists to pickle file
    with open('data.pickle', 'wb') as f:
        pickle.dump((phrases, labels, training), f)
    print('training:')
    print(training)
    print('output')
    print(output)

# build model
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)

if os.path.exists('gene.casual.meta'):
    model.load('gene.casual')
else:
    model.fit(training, output, n_epoch=50, batch_size=32, show_metric=True)
    model.save('gene.casual')

def bag_of_words(s, phrases):
    bag =[0 for _ in range(len(phrases))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(phrases):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chat():
    print('sup?')
    while True:
        inp = input("you: ")
        if inp.lower() == 'quit':
            break
        results = model.predict([bag_of_words(inp, phrases)])
        results_index = np.argmax(results)
        tag = labels[results_index]
        print(tag)

        for tg in data['data']:
            if tg['tag'] == tag:
                responses = tg['response']
        print(random.choice(responses))

chat()