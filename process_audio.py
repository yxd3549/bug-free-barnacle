import csv
import glob
import math
import numpy
import wave
from audiolazy.lazy_lpc import lpc
from scipy.signal import lfilter, hamming

K = 31
FRAME_DURATION = 16  # Milliseconds
AUDIO_WINDOW_DURATION = 520  # Milliseconds


def extract_features(input_path, output_path):
    for file in glob.iglob(input_path + '/**/*.wav', recursive=True):
        audio = wave.open(file, 'r')  # Open audio file
        audio_rate = audio.getframerate() // 1000  # In milliseconds
        x = audio.readframes(-1)  # Read entire file
        x = numpy.frombuffer(x, dtype='int16')  # Read as ints

        window_size = AUDIO_WINDOW_DURATION * audio_rate  # Calculate frames per window
        frame_size = FRAME_DURATION * audio_rate  # Calculate frames per... frame

        x = x[(audio_rate*1000)-(window_size//2):]  # Skip the 1st second of the video, but center audio
        n = 0
        for i in range(0, 1000*audio_rate, 100*audio_rate):  # Windows are 100ms apart, each 520ms in length
            window = x[i:i+window_size]
            with open(output_path + file[-25:-4] + '-' + str(n) + '.csv', 'w', newline='') as output_file:
                writer = csv.writer(output_file, delimiter=',')
                for j in range(0, window_size - frame_size//2, frame_size//2):  # Each 'frame' is 16ms in length
                    frame = window[j:j+frame_size]
                    formants = get_formants(frame, K)
                    writer.writerow(formants)
            n += 1
        break


def get_formants(x, k):
    # Get Hamming window.
    n = len(x)
    w = numpy.hamming(n)

    # Apply window and high pass filter
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Get LPC
    result_filter = lpc(x1, order=k)
    coefficients = result_filter.numerator

    return coefficients


if __name__ == '__main__':
    extract_features('data/audio_speech/', 'processed/audio/')
