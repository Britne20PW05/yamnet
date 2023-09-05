import csv
import io

import wave

import openpyxl
import pyaudio
import tensorflow as tf
import numpy as np

import pandas as pd

interpreter = tf.lite.Interpreter('lite-model_yamnet_tflite_1.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


class AudioFile:

    def __init__(self, file):
        """ Init audio stream """
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.wf.getsampwidth()),
            channels=self.wf.getnchannels(),
            rate=self.wf.getframerate(),
            output=True
        )

    def play(self):
        chunk = 16000
        """ Play entire file """
        while True:
            data = self.wf.readframes(chunk)
            if len(data) < chunk:
                break  # End of file reached

            waveform = np.frombuffer(data, dtype=np.int16) / 32767
            waveform1 = waveform.astype(np.float32)

            interpreter.resize_tensor_input(waveform_input_index, [len(waveform1)], strict=False)
            interpreter.allocate_tensors()
            interpreter.set_tensor(waveform_input_index, waveform1)
            interpreter.invoke()

            scores, embeddings, spectrogram = (
                interpreter.get_tensor(scores_output_index), interpreter.get_tensor(embeddings_output_index),
                interpreter.get_tensor(spectrogram_output_index))

            # print(scores.mean(axis=0).argmax())
            class_names = class_names_from_csv(open('yamnet_class_map.csv').read())
            print(class_names[scores.mean(axis=0).argmax()])
            #print('done')






# Usage example for pyaudio
a = AudioFile("music_0_speech_10_vehicle_12.wav")
a.play()






