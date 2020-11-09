import numpy
import wave
import math
from scipy.signal import lfilter, hamming
from audiolazy.lazy_lpc import lpc


def get_formants(filepath):
    audio = wave.open(filepath, 'r')

    # Get file as numpy array.
    x = audio.readframes(-1)
    x = numpy.frombuffer(x, dtype='int16')

    # Get Hamming window.
    n = len(x)
    w = numpy.hamming(n)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Get LPC.
    framerate = audio.getframerate()
    number_of_coefficients = 2 + framerate // 1000
    result_filter = lpc(x1, order=number_of_coefficients)
    coefficients = result_filter.numerator

    return coefficients


print(get_formants('./data/audio_song/Actor_01/03-02-01-01-01-01-01.wav'))
