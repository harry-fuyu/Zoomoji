from imutils import face_utils
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import numpy as np






def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[6])
    mar = (1.0)/(4.0*A)
    return mar


def classify(shape, mouth_thr=0.6, wink_thr=0.23):
    # determine mouth type
    left, right, top, down = shape[48], shape[54], shape[51], shape[57]
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(top - down)
    mouth_ratio = height / width

    if mouth_ratio > mouth_thr:  # o
        mouth = "o"
    else:
        mouth = "-"

    # determine eye
    l_left, l_right, l_top, l_down = shape[36], shape[39], shape[38], shape[40]
    r_left, r_right, r_top, r_down = shape[42], shape[45], shape[44], shape[46]   
    l_ratio = np.linalg.norm(l_top - l_down) / np.linalg.norm(l_left - l_right) 
    r_ratio = np.linalg.norm(r_top - r_down) / np.linalg.norm(r_left - r_right)

    if l_ratio < wink_thr:
        l_eye = "<"
    else:
        l_eye = "o"

    if r_ratio < wink_thr:
        r_eye = "<"
    else:
        r_eye = "o"
    
    return [mouth, l_eye, r_eye]
    
    

PATH = "/Users/kaiyinghou/Documents/Github/zoomoji/Zoomoji/"
p = PATH + "models/landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
cap = cv2.VideoCapture(0)

lar1 = 0
lar2 = 0
lar3 = 0

rar1 = 0
rar2 = 0
rar3 = 0




while True:
    # Getting out image by webcam 
    _, image = cap.read()

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    bg = cv2.imread(PATH + "/background/background.png")

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract points for different parts on the face
        face_cloud = dict()
        for key in face_utils.FACIAL_LANDMARKS_IDXS.keys():
            start, end = face_utils.FACIAL_LANDMARKS_IDXS[key]
            organ = shape[start:end]
            face_cloud[key] = organ

        # classification algorithm for each part
        mouth = classify(shape)[0]
        l_eye = classify(shape)[1]
        r_eye = classify(shape)[2]




        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)


        lar1 = lar2
        lar2 = lar3
        lar3 = leftEAR

        rar1 = rar2
        rar2 = rar3
        rar3 = rightEAR



        if (lar1 + lar2 + lar3) / 3 < 0.22:
            plt.text(shape[41,0], -1*shape[41,1]-25, "> blink")
        if (rar1 + rar2 + rar3) / 3 < 0.22:
            plt.text(shape[46,0], -1*shape[46,1]-25, "< blink")
            




        if mouth == 'o':
            plt.text(shape[8,0], -1*shape[8,1]-50, "O mouth")
        # if l_eye == '<':
        #     plt.text(shape[41,0], -1*shape[41,1]-25, "> eye")
        # if r_eye == '<':
        #     plt.text(shape[46,0], -1*shape[46,1]-25, "< eye")

        plt.scatter(shape[:,0], -1 * shape[:,1])
        




    plt.xlim(250, 1000)
    plt.ylim(-750, 0)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf() 
        
    




        
