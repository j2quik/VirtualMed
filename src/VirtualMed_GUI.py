###################################################################################################
# Author: James Hamilton
# All code contained in this module written by James Hamilton
#
# Module: VirtualMed_GUI.py
#
# Description:
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

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from tkinter import *



################################################################################
# Send user response to chat log, get VirtualMed response and log to chat
################################################################################
def send():
    message = text_box.get("1.0",'end-1c').strip()
    text_box.delete("0.0",END)

    if message != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "{}: ".format(user) + message + '\n\n')
        chat_log.config(foreground="#442265", font=("Arial", 12 ))
    
        response = get_response(message)
        chat_log.insert(END, "VirtualMed: " + response + '\n\n')
            
        chat_log.config(state=DISABLED)
        chat_log.yview(END)



################################################################################
# Get response to the user message
################################################################################
def get_response(message):
    predictions = predict_tag(message, model)
    response = evaluate_predictions(predictions, intents)
    
    return response



################################################################################
# Predict the type of message
################################################################################
def predict_tag(message, model):
    error_threshold = 0.25

    bag = bag_of_words(message, words)
    predictions = model.predict(np.array([bag]))[0]

    # filter out predictions below the threshold
    results = [[i,pred] for i, pred in enumerate(predictions) if pred > error_threshold]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    predictions = list()

    for result in results:
        predictions.append({"intent": tags[result[0]], "probability": str(result[1])})
    if len(predictions) == 0:
        predictions.append({"intent": "default", "probability": "1.0"})
    
    return predictions



################################################################################
# Return a 0 or 1 for each word in the bag that exists in the message
################################################################################
def bag_of_words(message, words):
    message = clean_message(message)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)

    for term in message:
        for i, word in enumerate(words):
            if word == term: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1

    return (np.array(bag))



################################################################################
# Split message into words and lemmatize each word
################################################################################
def clean_message(message):
    message = nltk.word_tokenize(message)
    message = [lemmatizer.lemmatize(word.lower()) for word in message]
    
    return message



################################################################################
# Determine the proper response based on predictions of the message
################################################################################
def evaluate_predictions(predictions, intents):
    tag = predictions[0]['intent']
    possible_replies = intents['intents']

    for reply in possible_replies:
        if(reply['tag'] == tag):
            response = random.choice(reply['responses'])
    
    return response



################################################################################
# Place all components on the GUI
################################################################################
def build_gui():
    # creates base GUI
    virtual_med = Tk()
    virtual_med.title("VirtualMed")
    virtual_med.geometry("720x480")
    virtual_med.resizable(width=TRUE, height=TRUE)

    # creates chat log window
    chat_log = Text(virtual_med, bd=0, bg="light cyan", font="Arial", wrap="word")
    chat_log.config()

    # bind scrollbar to chat window
    scrollbar = Scrollbar(virtual_med, command=chat_log.yview, cursor="heart")
    chat_log['yscrollcommand'] = scrollbar.set

    # creates text entry region
    text_box = Text(virtual_med, bd="0", bg="white", font="Arial", wrap="word")
    #text_box.bind("<Return>", send)

    # creates send button
    send_button = Button(virtual_med, font=("Arial", 12, 'bold'), text="Send", bd="0", bg="#32de97", 
                        activebackground="#3c9d9b", fg='#ffffff', command=send)

    return virtual_med, chat_log, scrollbar, text_box, send_button



################################################################################
# Display all components of the GUI
################################################################################
def display_gui():
    chat_log.place(x=10, y=10, height=390, width=700)
    scrollbar.place(x=695, y=10, height=390, width=15)
    text_box.place(x=10, y=410, height=60, width=630)
    send_button.place(x=650, y=410, height=60, width=60)



####################################################################################################
# Entry point
####################################################################################################
if __name__ == "__main__":

    model = load_model('virtualmed_model.h5')
    intents = json.loads(open('intents_virtualmed.json').read())
    words = pickle.load(open('words_virtualmed.pkl','rb'))
    tags = pickle.load(open('tags_virtualmed.pkl','rb'))
    lemmatizer = WordNetLemmatizer()
    user = 'user'
    welcome = "Welcome! You can have a conversation with me about any health-related concerns because I am available anytime. Let's get started..."
    ignore_words = ["'s", ",", ".", "?"]
    

    virtual_med, chat_log, scrollbar, text_box, send_button = build_gui()
    display_gui()
    chat_log.insert(END, "VirtualMed: {}\n\n".format(welcome))

    virtual_med.mainloop()
