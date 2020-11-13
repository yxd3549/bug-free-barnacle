import csv
import glob
import math
import numpy
import wave
from audiolazy.lazy_lpc import lpc
from scipy.signal import lfilter, hamming


def extract_features(input_path, output_path):
    for file in glob.iglob(input_path + '/**/*.wav', recursive=True):
        with open(output_path + file[-25:-4] + '.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=',')
            audio = wave.open(file, 'r')
            frame_rate = audio.getframerate()
            n_frames = audio.getnframes()
            x = audio.readframes(-1)
            x = numpy.frombuffer(x, dtype='int16')
            for i in range(0, n_frames, frame_rate):
                formants = get_formants(x[i:i+frame_rate], frame_rate)
                writer.writerow(formants)


def get_formants(x, framerate):
    # Get Hamming window.
    n = len(x)
    w = numpy.hamming(n)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Get LPC.
    number_of_coefficients = 2 + framerate // 1000
    result_filter = lpc(x1, order=number_of_coefficients)
    coefficients = result_filter.numerator

    return coefficients


if __name__ == '__main__':
    extract_features('data/', 'processed/audio/')