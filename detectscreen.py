import argparse
import cv2
import numpy as np

def contoursConvexHull(contours):
    pts = []
    for i in range(0, len(contours)):
        for j in range(0, len(contours[i])):
            pts.append(contours[i][j])

    pts = np.array(pts)

    result = cv2.convexHull(pts)

    return result

def process_video(path_to_video):
    """Process the given video"""

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise Exception("oi why cant i open your video".format(v=path_to_video))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 100)
        edgesblurred = cv2.Canny(blurred, 50, 100)

        # cv2.imshow('fram', edgesblurred)

        contours, hierarchy = cv2.findContours(edgesblurred,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        ConvexHullPoints = contoursConvexHull(contours)
        cv2.polylines(frame, [ConvexHullPoints], True, (0, 0, 255), 3)

        # cv2.imshow('f', frame)

        somerandomshit(frame)

        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def somerandomshit(frame):

    def clahe(img, clip_limit=2.0, grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(img)

    hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))

    # Adaptive Thresholding to isolate the bed
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 2)

    contours, hierarchy = cv2.findContours(img_th,
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

    # Filter the rectangle by choosing only the big ones
    # and choose the brightest rectangle as the bed
    max_brightness = 0
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w * h > 40000 and w * h < 250000:
            mask = np.zeros(frame.shape, np.uint8)
            mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]

            cv2.imshow('mask', mask)
            if cv2.waitKey(30) == ord("q"):
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process the given video")
    parser.add_argument("path_to_video")
    args = parser.parse_args()
    process_video(args.path_to_video)

