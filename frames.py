import cv2
import glob
from pathlib import Path


def extract_frames(path, out):
    for file in glob.iglob(path + '/**/*.mp4', recursive=True):
        vidcap = cv2.VideoCapture(file)
        success, image = vidcap.read()
        count = 0
        success = True
        path = Path(out + '/' + file[:-4])
        path.mkdir(parents=True, exist_ok=True)
        while success:
            cv2.imwrite(str(path) + "/frame_%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1


if __name__ == '__main__':
    path = input("path: ")
    out = 'frames'
    extract_frames(path, out)
