# chatbot.py

import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


print("Starting SVREC Chatbot...")

lemmatizer = WordNetLemmatizer()

# Load files
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)

    result = model.predict(
        np.array([bow]),
        verbose=0
    )[0]

    ERROR_THRESHOLD = 0.25

    results = []

    for i, r in enumerate(result):
        if r > ERROR_THRESHOLD:
            results.append((i, r))

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append(classes[r[0]])

    return return_list


def get_response(intents_list):
    if len(intents_list) == 0:
        return "Sorry, I did not understand that. Please ask again."

    tag = intents_list[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I am not sure about that."


# Chat loop
print("SVREC Chatbot is ready!")
print("Type 'quit' to exit.\n")

while True:

    message = input("You: ")

    if message.lower() == "quit":
        print("Bot: Thank you! Have a nice day.")
        break

    ints = predict_class(message)

    reply = get_response(ints)

    print("Bot:", reply)
