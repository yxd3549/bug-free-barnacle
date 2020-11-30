import csv
import glob
from pathlib import Path
import numpy as np
import cv2

from landmarks import detect_face, generate_landmarks
from mask import draw_mask

AUDIO_PATH = 'processed/audio/'
FRAMES_PATH = 'processed/frames/'
OUT_PATH = 'processed/landmarks/'

def generate_features(audio=True, frames=True):

    out_path = Path(OUT_PATH + 'unmasked/')
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(OUT_PATH + 'masked/')
    out_path.mkdir(parents=True, exist_ok=True)

    # generate audio features
    if (audio):
        extract_features('data/audio_speech/', AUDIO_PATH)

    # generate frames
    if (frames):
        extract_frames('data/video_speech/', VIDEO_PATH)

    # generate landmarks for each frame
    for file in glob.iglob(FRAMES_PATH + '*.jpg'):

        filename = Path(file).name[:-4]

        # find correct landmarks
        original = cv2.imread(file)
        rect = detect_face(original)
        shape = generate_landmarks(original, rect)

        # save
        with open(OUT_PATH + 'unmasked/' + filename + '.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=',')
            for i in range(len(shape)):
                writer.writerow(shape[i])

        # find masked landmarks
        draw_mask(file, 'temp.jpg')
        masked = cv2.imread('temp.jpg')
        shape = generate_landmarks(masked, rect)

        # save
        with open(OUT_PATH + 'masked/' + filename + '.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=',')
            for i in range(len(shape)):
                writer.writerow(shape[i])


generate_features(False, False)
