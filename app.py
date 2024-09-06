from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load your model and tokenizer
model = tf.keras.models.load_model('Side projects/Cyber_bullying_detection_model.keras')

with open('Side projects/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100  # Example max length, replace with your actual max_len

app = Flask(__name__)

def predict_offensive(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return 'Offensive' if prediction[0][1] > 0.5 else 'Not Offensive'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def check():
    text = request.form['text']
    result = predict_offensive(text)
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
