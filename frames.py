import cv2
import glob
from pathlib import Path


def extract_frames(path, out):
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    for file in glob.iglob(path + '/**/*.mp4', recursive=True):
        file_path = Path(file)
        if file_path.name[:2] == '02':
            continue


        vidcap = cv2.VideoCapture(file)
        # success, image = vidcap.read()
        count = 0
        frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS))
        # length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Skip the first second
        for _ in range(frame_rate):
            vidcap.read()

        # number of windows
        for _ in range(10):
            success, image = vidcap.read()
            cv2.imwrite(out + '/' + file_path.name[:-4] + "-frame-%d.jpg" % count, image)  # save frame as JPEG file
            count += 1
            for _ in range(frame_rate//10):
               vidcap.read()



if __name__ == '__main__':
    path = input("path: ")
    out = 'frames'
    extract_frames(path, out)
