# train.py
print("Training started...")

import nltk
import json
import random
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Download required NLTK data (only first time)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents file
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

words = []
classes = []
documents = []

ignore_letters = ['?', '!', '.', ',']

# Process intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # Tokenize sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # Add to documents
        documents.append((word_list, intent["tag"]))

        # Add class
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and clean words
words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_letters
]

words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Training data
training = []
output_empty = [0] * len(classes)

for doc in documents:

    bag = []

    pattern_words = [
        lemmatizer.lemmatize(word.lower())
        for word in doc[0]
    ]

    for word in words:
        bag.append(1 if word in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle data
random.shuffle(training)

training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build Neural Network
model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train model
model.fit(
    np.array(train_x),
    np.array(train_y),
    epochs=200,
    batch_size=5,
    verbose=1
)

# Save model
model.save("chatbot_model.h5")

print("Training completed successfully!")
print("Files created:")
print("- words.pkl")
print("- classes.pkl")
print("- chatbot_model.h5")

