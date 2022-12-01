###################################################################################################
# James Hamilton
#
# VirtualMed_TRAIN.py
#
# This application opens a message/response GUI.
# You can type a question in the message window and 
# get a response for your health-related concerns.
###################################################################################################



import warnings
import os
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import random
import matplotlib.pyplot as plt



####################################################################################################
# Get the words, patterns and tags from the database
####################################################################################################
def get_data():
    patterns_with_tags = list()
    words = list()
    tags = list()
    count = 0

    for intent in intents['intents']:
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            if word not in ignore_words:
                words.extend(word)
            else:
                print("excluded word:", word)
            patterns_with_tags.append((word, intent['tag']))
            count = count +1
    
    print('count:', count)
    return patterns_with_tags, words, tags



####################################################################################################
# Save the list of words and tags to pickle and text files for use with GUI
####################################################################################################
def save_data():
    pickle.dump(words,open('words_virtualmed.pkl','wb'))
    with open("words_virtualmed.txt", "w") as output:
        output.write(str(words))
    output.close()

    pickle.dump(tags,open('tags_virtualmed.pkl','wb'))
    with open("tags_virtualmed.txt", "w") as output:
        output.write(str(tags))
    output.close()



####################################################################################################
# Define function to handle tokenization
####################################################################################################
def get_training_set():
    training_set = list()
    
    empty_arr = [0] * len(tags)  # create an empty array for our output

    # training set, bag of words for each sentence
    for pattern_with_tag in patterns_with_tags:
        bag = list()
        pattern = pattern_with_tag[0]  # list of tokenized words for the pattern
        # lemmatize each word - create base word, in attempt to represent related words
        pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]
        
        # create our bag of words array with 1, if word match found in current pattern
        for word in words:
            bag.append(1) if word in pattern else bag.append(0)

        output_row = list(empty_arr)  # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row[tags.index(pattern_with_tag[1])] = 1
        
        training_set.append([bag, output_row])
        
    random.shuffle(training_set)
    training_set = np.array(training_set)
    training_patterns = list(training_set[:,0])
    training_tags = list(training_set[:,1])


    return training_patterns, training_tags



    # create our training data
    #training_set = list()
    training_patterns = list()
    training_tags = list()
    # create an empty array for our output
    empty_arr = [0] * len(tags)

    # training set, bag of words for each sentence
    for pattern_with_tag in patterns_with_tags:
        #bag = list()
        training_pattern = list()
        training_tag = list()
        # list of tokenized words for the pattern
        pattern = pattern_with_tag[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]
        # create our bag of words array with 1, if word match found in current pattern
        for word in words:
            #bag.append(1) if word in pattern else bag.append(0)
            training_pattern.append(1) if word in pattern else training_pattern.append(0)
        # output is a '0' for each tag and '1' for current tag (for each pattern)
        #output_row = list(output_empty)
        training_tag = list(empty_arr)
        training_tag[tags.index(pattern_with_tag[1])] = 1
        
        #training_set.append([bag, output_row])
        training_patterns.append(training_pattern)
        training_tags.append(training_tag)

    #random.shuffle(training_set)
    #training_set = np.array(training_set)
    #training_patterns = list(training_set[:,0])
    #training_tags = list(training_set[:,1])


    return training_patterns, training_tags



####################################################################################################
# Build the model
####################################################################################################
def build_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(len(training_patterns[0]),), activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(32768, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=[acc])
    model.summary()

    return model



####################################################################################################
# Plot the training results
####################################################################################################
def plot_results(title, num_epochs, train, val, id):
    plt.figure()
    N = np.arange(0, num_epochs)
    plt.plot(N, train, label="Training {}".format(title))
    plt.plot(N, val, label="Training {}".format(title))
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.legend(loc="best")
    plt.savefig('plots/{}_epochs={}_{}.png'.format(title, num_epochs, id))



####################################################################################################
# Entry point
####################################################################################################

if __name__=='__main__':

    id = sys.argv[1]  # get the first argument & store it to append the model name
    num_epochs = 100
    batch_size = 8
    learning_rate = 0.0001
    epsilon = 0.001

    #data_file_name = 'intents_virtualmed.json'
    filepath = 'train_weights/best_weights_{}.h5'.format(id)  # name of file to save best weights during training
    model_name = 'models/virtualmed_model_{}.h5'.format(id)   # name of model saved after training
    lemmatizer = WordNetLemmatizer()
    optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon)
    loss = CategoricalCrossentropy()
    acc = CategoricalAccuracy('accuracy')
    ignore_words = ["'s", ",", ".", "?"]
    
    data_file = open('intents_virtualmed.json').read()
    intents = json.loads(data_file)

    patterns_with_tags, words, tags = get_data()

    num_classes = len(tags)
    print(len(words), "words")

    # lemmaztize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]

    save_data()

    training_patterns, training_tags = get_training_set()

    print (len(tags), "tags")
    print (len(words), "lemmatized words")
    print(len(training_patterns[0]), "training patterns")
    print("\nTraining data created")

    model = build_model()

    # add a checkpoint to save the weights with the lowest validation loss
    best_weights = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)

    results = model.fit(np.array(training_patterns), np.array(training_tags), epochs=num_epochs, batch_size=batch_size, validation_split=0.2, callbacks=[best_weights], verbose=1)

    model.save(model_name, results)

    train_loss = results.history['loss']
    train_acc  = results.history['accuracy']
    val_loss = results.history['val_loss']
    val_acc  = results.history['val_accuracy']

    print('\n######################################## TRAINING RESULTS ########################################')
    print('\nbest train loss: {:.5f}'.format(min(train_loss)))
    print('best train acc: {:.4f}'.format(max(train_acc)))
    print('\nbest val loss: {:.5f}'.format(min(val_loss)))
    print('best val acc: {:.4f}'.format(max(val_acc)))

    plot_results('Loss', num_epochs, train_loss, val_loss, id)
    plot_results('Accuracy', num_epochs, train_acc, val_acc, id)

    print("\nmodel created")
