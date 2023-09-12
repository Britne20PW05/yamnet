import csv
import io
import wave

import librosa
import pyaudio
import tensorflow as tf
import numpy as np
import resampy

interpreter = tf.lite.Interpreter('lite-model.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
#embeddings_output_index = output_details[1]['index']
#spectrogram_output_index = output_details[2]['index']

print(scores_output_index)

# Set a fixed number of MFCC coefficients for all audio files
num_mfcc = 20

# Set a smaller maximum length for padding/truncating MFCC sequences
max_length = 36


def extract_mfcc(audio_file_path, num_mfcc=44, target_sample_rate=16000, desired_length=2048):
    audio_data, sample_rate = librosa.load(audio_file_path, sr=target_sample_rate, duration=10.00, res_type='kaiser_fast')

    # Zero-pad the audio signal if it's shorter than desired_length
    if len(audio_data) < desired_length:
        shortage = desired_length - len(audio_data)
        padding = shortage // 2
        audio_data = np.pad(audio_data, (padding, shortage - padding), 'constant')

    # Resample the audio to the target sample rate
    audio_data = resampy.resample(audio_data, sample_rate, target_sample_rate)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=num_mfcc)

    return mfccs

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


class AudioFile:

    def __init__(self, file):

        '''mfccs = extract_mfcc(file, num_mfcc=num_mfcc)

        # Pad or truncate the MFCC sequence to the maximum length
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]

        # Add the channel dimension
        mfccs = np.expand_dims(mfccs, axis=-1)

        # Reshape mfccs to match the expected shape (batch_size, max_length, num_mfcc, 1)
        mfccs = mfccs.reshape(1, max_length, num_mfcc, 1)'''


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

            '''spectogram = tf.signal.stft(waveform1, frame_length=255, frame_step=128)
            spectogram= tf.abs(spectogram)
            spectogram = spectogram[..., tf.newaxis]
            spectogram

            waveform1=spectogram'''



            interpreter.resize_tensor_input(waveform_input_index, [len(waveform1)], strict=False)
            interpreter.allocate_tensors()
            interpreter.set_tensor(waveform_input_index, waveform1)
            interpreter.invoke()

            scores = (interpreter.get_tensor(scores_output_index))

            # print(scores.mean(axis=0).argmax())
            class_names = class_names_from_csv(open('yamnet_class_map.csv').read())
            print(class_names[scores.mean(axis=0).argmax()])
            #print('done')

# Usage example for pyaudio
a = AudioFile(r"C:\Britne\folder1\AudioAlert\Dataset\Cat\cat_4.wav")
a.play()





