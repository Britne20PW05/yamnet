import wave

import pyaudio
import threading
import numpy as np
import scipy.signal as signal
import tensorflow as tf
import csv
import io

# Constants for audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Stereo audio
RATE = 44100  # Sample rate (samples per second)
CHUNK = 4410  # Size of each audio chunk (frames per buffer)
RECORD_SECONDS = 5  # Duration of recording in seconds

# Create a PyAudio object
audio = pyaudio.PyAudio()
# Create an event to signal the playback thread
playback_event = threading.Event()
audio_ipdata=[]
audio_opdata=[]
FRAMES=10

interpreter = tf.lite.Interpreter('lite-model.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names

def asc_prediction(data):
    interpreter.resize_tensor_input(waveform_input_index, [len(data)], strict=False)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, data)
    interpreter.invoke()
    scores = interpreter.get_tensor(scores_output_index)
    class_names = class_names_from_csv(open('yamnet_class_map.csv').read())
    elements = (class_names[scores.mean(axis=0).argmax()])
    print(elements)


def doa(audio_data):
    global audio_opdata
    audio_ipdata.append(np.frombuffer(audio_data, dtype=np.int16))
    if (len(audio_ipdata) == FRAMES):
        audio_array = np.concatenate(audio_ipdata)
        resampled_audio = signal.resample(audio_array, int(len(audio_array) * 16000 / 44100))
        resampled_audio = resampled_audio.astype(audio_array.dtype)
        resampled_audio = resampled_audio / 32767
        data = np.float32(resampled_audio)
        #print(len(data))
        asc_prediction(data)
        audio_ipdata.clear()

# Define a callback function for audio processing
def audio_callback(in_data, frame_count, time_info, status):
    # You can process the audio data here
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    # For example, you can print the shape of the audio data
    #print(f"Received audio data shape: {audio_data.shape}")
    doa(audio_data)
    return in_data, pyaudio.paContinue

# Open an audio stream with the specified parameters and callback function
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)
# Start the audio stream
stream.start_stream()


# Record for the specified duration
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:

    print("Recording stopped by user.")

# Wait for the playback thread to finish

# Stop and close the audio stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio instance
audio.terminate()

print("Recording finished.")
