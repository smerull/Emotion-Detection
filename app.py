import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer dan model
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('model/lstm_model.h5', compile=False)
max_len = 128
labels = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Fungsi untuk preprocessing input
def prepare_input(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded

# UI
st.title("Emotion Classification Web App")
st.write("Masukkan kalimat dalam bahasa Inggris untuk mengklasifikasikan emosi.")

user_input = st.text_area("Input Kalimat", "")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan kalimat terlebih dahulu.")
    else:
        input_seq = prepare_input(user_input)
        probs = model.predict(input_seq)[0]
        predicted_label = labels[np.argmax(probs)]
        confidence = np.max(probs) * 100

        st.subheader("Hasil Prediksi")
        st.write(f"**Emosi:** {predicted_label}")
        st.write(f"**Kepercayaan:** {confidence:.2f}%")

        # Opsional: tampilkan distribusi probabilitas
        st.bar_chart({label: prob for label, prob in zip(labels, probs)})
