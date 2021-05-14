import cv2
import dlib
import numpy as np
from math import hypot

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def nothing(x):
    pass

def midpoint(p1,p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def blinking(eye_outline, eye_landmarks):
    left_extreme = (eye_landmarks.part(eye_outline[0]).x, eye_landmarks.part(eye_outline[0]).y)
    right_extreme = (eye_landmarks.part(eye_outline[3]).x, eye_landmarks.part(eye_outline[3]).y)
    top_extreme = midpoint(eye_landmarks.part(eye_outline[1]), eye_landmarks.part(eye_outline[2]))
    bottom_extreme = midpoint(eye_landmarks.part(eye_outline[5]), eye_landmarks.part(eye_outline[4]))
    horiline_len = hypot((left_extreme[0] - right_extreme[0]), (left_extreme[1] - right_extreme[1]))
    vertiline_len = hypot((top_extreme[0] - bottom_extreme[0]), (top_extreme[1] - bottom_extreme[1]))
    ratio = horiline_len / vertiline_len
    return ratio
def gaze(eye_outlines, face_landmarks):
    eye_region = np.array([(face_landmarks.part(eye_outlines[0]).x, face_landmarks.part(eye_outlines[0]).y),
                           (face_landmarks.part(eye_outlines[1]).x, face_landmarks.part(eye_outlines[1]).y),
                           (face_landmarks.part(eye_outlines[2]).x, face_landmarks.part(eye_outlines[2]).y),
                           (face_landmarks.part(eye_outlines[3]).x, face_landmarks.part(eye_outlines[3]).y),
                           (face_landmarks.part(eye_outlines[4]).x, face_landmarks.part(eye_outlines[4]).y),
                           (face_landmarks.part(eye_outlines[5]).x, face_landmarks.part(eye_outlines[5]).y)], np.int32)
    h, w, _ = img.shape
    black_out = np.zeros((h, w), np.uint8)
    cv2.polylines(black_out, [eye_region], True, 255, 2)
    cv2.fillPoly(black_out, [eye_region], 255)
    eye = cv2.bitwise_and(gray_image, gray_image, mask=black_out)
    minx = np.min(eye_region[:, 0])
    maxx = np.max(eye_region[:, 0])
    miny = np.min(eye_region[:, 1])
    maxy = np.max(eye_region[:, 1])
    gray_eye = eye[miny: maxy, minx: maxx]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    ratio1=0
    try:
        ratio1 = left_side_white/right_side_white
    except ZeroDivisionError:
        print("e")
    return ratio1

def black_out(mask, eye_landmarks):
    points = [shape[i] for i in eye_landmarks]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def npcoversion(shape, dtype="int"):
    xyvalues = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        xyvalues[i] = (shape.part(i).x, shape.part(i).y)
    return xyvalues

def contours(thresh, mid, img, right=False):
    contouring, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        contourings = max(contouring, key=cv2.contourArea)
        matrix = cv2.moments(contourings)
        p1 = int(matrix['m10'] / matrix['m00'])
        p2 = int(matrix['m01'] / matrix['m00'])
        if right:
            p1 += mid
        cv2.circle(img, (p1, p2), 4, (180, 255 , 150), 2)
    except:
        return 0
image = cv2.VideoCapture(0)
ret, img = image.read()
thresh = img.copy()
kernel = np.ones((9, 9), np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

while (True):
    ret, img = image.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image, 1)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0,0, 255), 2)
        shape = predictor(gray_image, face)
        total_blink_ratio = (blinking([36, 37, 38, 39, 40, 41], shape) + blinking([42, 43, 44, 45, 46, 47],
                                                                                      shape)) / 2
        gaze_ratio_left_eye = gaze([36, 37, 38, 39, 40, 41], shape)
        gaze_ratio_right_eye = gaze([42, 43, 44, 45, 46, 47], shape)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        shape = npcoversion(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = black_out(mask, [36, 37, 38, 39, 40, 41])
        mask = black_out(mask, [42, 43, 44, 45, 46, 47])
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        gray_eyes = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold','image')
        _, thresh = cv2.threshold(gray_eyes, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)  # 1
        thresh = cv2.dilate(thresh, None, iterations=4)  # 2
        thresh = cv2.medianBlur(thresh, 3)  # 3
        thresh = cv2.bitwise_not(thresh)
        contours(thresh[:, 0:mid], mid, img)
        contours(thresh[:, mid:], mid, img, True)
        if total_blink_ratio > 5.5:
            cv2.putText(img, "BLINKING", (50, 50), cv2.FONT_ITALIC, 2, (0, 0 , 0),3)
        if gaze_ratio <= 0.45:
            cv2.putText(img, "RIGHT", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)
        elif gaze_ratio > 1.8:
            cv2.putText(img, "LEFT", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 225, 255), 3)
        else:
            cv2.putText(img, "CENTER", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)
    cv2.imshow('Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

image.release()
cv2.destroyAllWindows()






