import tkinter as tk
from tkinter import filedialog

import joblib
import numpy as np
from scipy.io import wavfile

from constants import CHUNK_SIZE, INSTRUMENTS
import sounddevice as sd

window = tk.Tk()

window.title("My Window")
window.geometry("500x300")

error_text = tk.StringVar()
error_text.set('')

error_label = tk.Label(window, textvariable=error_text, bg='white', font=('Arial', 12), width=30, height=2)
error_label.place(x=0, y=280, anchor='sw')
error_label.pack()

MODEL = joblib.load('ddd.pkl')
DECISION_WINDOW = 15

print(MODEL)


def choose_file():
    filename = filedialog.askopenfilename()
    if not filename:
        error_text.set('No file chosen')
        return
    if not filename.endswith('.wav'):
        error_text.set('File must be .wav')
        return
    error_text.set('')
    fs, data = wavfile.read(filename)
    answers = []
    for chunk_id in range(len(data) // CHUNK_SIZE):
        chunk = data[chunk_id * CHUNK_SIZE:(chunk_id + 1) * CHUNK_SIZE]
        chunk = chunk.astype(np.float64)
        chunk /= 32768
        chunk /= 14

        chunk_fft = np.fft.fft(chunk)[:CHUNK_SIZE // 2]
        chunk_fft = np.abs(chunk_fft)

        answers.append(MODEL.predict(np.array([chunk_fft]))[0])


    decision_sum = np.zeros(len(INSTRUMENTS))

    # play sound and get which second of sample now

    sd.play(data, fs)

    # get current second of sample

    begin = sd.get_stream().time
    results = []
    for i in range(len(answers)):
        instruments = []
        decision_sum += answers[i]
        if i >= DECISION_WINDOW:
            decision_sum -= answers[i - DECISION_WINDOW]
        for i in range(len(INSTRUMENTS)):
            if decision_sum[i] >= DECISION_WINDOW / 2:
                instruments.append(INSTRUMENTS[i])
        results.append(" ".join(instruments))
        print(results[-1])


choose_file_button = tk.Button(window, text='Choose file', width=15, height=2, command=choose_file)

choose_file_button.pack()

window.mainloop()
