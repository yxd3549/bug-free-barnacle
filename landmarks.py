import numpy as np
import dlib
import cv2

def detect_face(image):

    # preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get face detector
    detector = dlib.get_frontal_face_detector()

    # detect faces
    rects = detector(image, 1)

    if (len(rects) < 1):
        print('No face detected')
        exit()

    # just grab the first face
    return rects[0]


def generate_landmarks(image, rect):

    # preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initialize face landmark predictor
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    # get predicted landmarks
    landmarks = predictor(image, rect)

    # read landmark object into np array of xy coordinates
    coords = np.zeros((68, 2), dtype='int')

    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    return coords


def visualize_landmarks(image_path, rect=None, display=True, output_path=None):

    # load image from disk
    image = cv2.imread(image_path)

    # detect face and landmarks
    if (rect is None):
        rect = detect_face(image)
    shape = generate_landmarks(image, rect)

    # add markers to image
    for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    # save to disk
    if not (output_path is None):
        cv2.imwrite(output_path, image)
    
    # display
    if (display):
        cv2.imshow("Output", image)
        cv2.waitKey(0)


visualize_landmarks('01-01-07-01-01-01-01-frame-9.jpg', None, False, '01-01-07-01-01-01-01-frame-9-landmarks.jpg')