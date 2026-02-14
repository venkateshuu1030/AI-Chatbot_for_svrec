from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import nltk
from keras.models import load_model

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Load trained files
model = load_model("chatbot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

import json
intents = json.loads(open("intents.json").read())


# Preprocess
def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "prob": str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Sorry, I didn't understand."

    tag = intents_list[0]["intent"]

    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return np.random.choice(i["responses"])


# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]

    ints = predict_class(msg)
    res = get_response(ints, intents)

    return jsonify({"reply": res})


if __name__ == "__main__":
    app.run(debug=True)
