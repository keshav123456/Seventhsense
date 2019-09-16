import argparse
import cv2
import numpy as np

def display(image, name, pos_x, pos_y):

    cv2.namedWindow(name)
    cv2.moveWindow(name, pos_x, pos_y)
    cv2.imshow(name, image)


def dw(frame):


    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([200,200,200], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(frame, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)



    return(res)



def process_video(path_to_video):
    """Process the given video"""

    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise Exception("oi why cant i open your video".format(v=path_to_video))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display(hsvframe, "hsv", 0, 0)
        display(frame, "show video", 1000,0)
        display(dw(frame), "white", 500, 0)

        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process the given video")
    parser.add_argument("path_to_video")
    args = parser.parse_args()
    process_video(args.path_to_video)
