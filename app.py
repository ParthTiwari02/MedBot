import json
from flask import Flask, jsonify, request
import pickle
import numpy
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import LancasterStemmer


with open("chatbot.pickle", "rb") as file:
    words, labels, training, output = pickle.load(file)

with open("intents.json") as file:
    data = json.load(file)

nltk.download('punkt')

stemmer = LancasterStemmer()

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

app = Flask(__name__)

@app.route("/chat/",methods=["POST"])
def chat_with_bot():
    query = request.get_json(force=True)
    Question = query["Question"]
    model = load_model("chatbotmodel.hdf5")
    currentText = bag_of_words(Question, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)
    Response = model.predict(numpyCurrentText[0:1])
    result_index = numpy.argmax(Response)
    tag = labels[result_index]

    if Response[0][result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                Response = tg['responses']
    else:
        Response = "Sorry, I didn't understand that."
    return {"Response" : Response}


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
