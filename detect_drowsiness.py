# Import necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
import playsound

# Function to play alarm sound asynchronously
def sound_alarm(path):
    playsound.playsound(path)

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute distance between the horizontal landmarks
    C = dist.euclidean(eye[0], eye[3])
    
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear

# Argument parser
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                    help="Path to facial landmark predictor")
    ap.add_argument("-a", "--alarm", type=str, default="alarm.wav",
                    help="Path to alarm .WAV file")
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="Index of webcam on system")
    return vars(ap.parse_args())

# Main drowsiness detection function
def drowsiness_detection(args):
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 48
    COUNTER = 0
    ALARM_ON = False
    
    # Load dlib's face detector and facial landmark predictor
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # Facial landmark indexes for eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Start the video stream
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            while True:
                # Grab frame, resize, and convert to grayscale
                frame = vs.read()
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in grayscale frame
                rects = detector(gray, 0)

                # Loop through face detections
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # Extract left and right eye, calculate EAR
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    # Draw contours for eyes
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # If EAR is below the threshold, increment counter
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1

                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            if not ALARM_ON:
                                ALARM_ON = True

                                # Play alarm asynchronously
                                if args["alarm"] != "":
                                    executor.submit(sound_alarm, args["alarm"])

                            # Display drowsiness alert
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        COUNTER = 0
                        ALARM_ON = False

                    # Display EAR value
                    cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display video stream
                cv2.imshow("Drowsiness Detection", frame)
                key = cv2.waitKey(1) & 0xFF

                # If `q` key is pressed, exit loop
                if key == ord("q"):
                    break
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            # Cleanup and stop video stream
            print("[INFO] Cleaning up...")
            vs.stop()
            cv2.destroyAllWindows()
            executor.shutdown(wait=False)

if __name__ == "__main__":
    try:
        args = parse_arguments()
        drowsiness_detection(args)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        # Ensure proper cleanup on exit
        print("[INFO] Exiting...")
        cv2.destroyAllWindows()
