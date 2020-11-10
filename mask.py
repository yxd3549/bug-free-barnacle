import numpy as np
import cv2
from landmarks import detect_face, generate_landmarks

def draw_mask(image_path, output_path):

    # generate landmarks
    image = cv2.imread(image_path)
    rect = detect_face(image)
    landmarks = generate_landmarks(image, rect)

    # get vertices of mask polygon from landmarks
    # pts 0-16 = jaw contour left to right
    # pts 27-30 = nose bridge top to bottom
    jaw = landmarks[2:15,:]
    n = jaw - landmarks[28,:]  # get approx normal of jaw
    jaw = np.add(jaw, n * 0.03)  # extrude slightly
    jaw = np.int32(jaw)
    mask = np.append(jaw, [landmarks[28,:]], axis=0)

    # load image and add mask
    image = cv2.imread(image_path)
    image = cv2.fillPoly(image, [mask], 0)

    # save masked image
    cv2.imwrite(output_path, image)
