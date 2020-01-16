# USAGE
# python yellowbox_dl.py --conf config/config.json

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
from imutils.io import TempFile
from imutils.video import FPS
from datetime import datetime
from threading import Thread
import numpy as np
import pandas as pd
import argparse
import dropbox
import imutils
import dlib
import time
import cv2
import os
import json
import csv
import math
import joblib
from skimage import filters
import matplotlib.pyplot as plt


def saveInJSON(carID, time):
    carDict = {
        "carID": carID,
        "TimeInYellowBox": time
    }
    carJSON = json.dumps(carDict)
    return carJSON


def verifyStoppingAnomaly(json, model):
    model = joblib.load(model)
    data = pd.read_json(json, typ='series')
    getTime = data['TimeInYellowBox']
    output = model.predict([[getTime]])
    return output


def testModel(carID, TimeInYellowBox, model):
    if math.isnan(TimeInYellowBox):
        print("Invalid TimeInYellowBox")
    else:
        myJSON = saveInJSON(carID, TimeInYellowBox)
        test = verifyStoppingAnomaly(myJSON, model)
        if test == 0:
            print('Anomaly CarID: ' + str(carID))
        elif test == 1:
            print('Not anomaly CarID ' + str(carID))
        else:
            print('Prediction error')


def findYellowBoxDeschartes(frame):
    cv2.line(frame, (400, 700), (400, 730), (0, 255, 0), 5)
    cv2.line(frame, (725, 700), (725, 730), (0, 255, 0), 5)
    cv2.line(frame, (745, 525), (745, 555), (0, 255, 0), 5)
    cv2.line(frame, (775, 425), (775, 455), (0, 255, 0), 5)
    cv2.line(frame, (355, 525), (355, 555), (0, 255, 0), 5)
    cv2.line(frame, (910, 525), (910, 555), (0, 255, 0), 5)
    cv2.line(frame, (575, 525), (575, 555), (0, 255, 0), 5)


def processFirstFrame(image):
    # Converting the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resize = imutils.resize(gray,height=400)

    # Smoothing without removing edges.
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    gray_filtered_resize = imutils.resize(gray_filtered,height=400)

    # Applying the canny filter
    edges = cv2.Canny(gray, 60, 120)
    edges_filtered = cv2.Canny(gray_filtered, 60, 120)
    edges_filtered_resize = imutils.resize(edges_filtered,height=400)

    
    
    cv2.imshow('Grayscaling',gray_resize)
    cv2.imshow('Grayscaling + filtering',gray_filtered_resize)
    cv2.imshow('Grayscaling + edge detection + filtering',edges_filtered_resize)


   


def findYellowBoxCasernes(frame):
    cv2.line(frame, (900, 735), (900, 700), (0, 255, 0), 5)
    cv2.line(frame, (950, 850), (950, 815), (0, 255, 0), 5)
    cv2.line(frame, (1000, 990), (1000, 955), (0, 255, 0), 5)


def saveInCsv(carID, timeInYellowBox, csvFile):
    rowToInsert = [carID, timeInYellowBox]
    existingLines = []
    with open(csvFile, 'r') as f:
        filereader = csv.reader(f, delimiter=' ')
        existingLines = [line for line in filereader]
    with open(csvFile, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # if sum(1 for row in filereader) > 0 :
        if rowToInsert not in existingLines:
            filewriter.writerow(rowToInsert)


def addHeader(csvFile):
    # with open('CSVFiles/casernes.csv',newline='') as f:
    #    r = csv.reader(f)
    #    data = [line for line in r]
    with open(csvFile, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['CarID', 'TimeInYellowBox'])
    # w.writerows(data)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="Path to the input configuration file")
ap.add_argument("-v", "--video", required=True, help="Path of video file to extract data from")
ap.add_argument("-f", "--csv", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# get the video file path
video_path = args["video"]

# get csv file name
csv_name = args["csv"]

# get model file name
model_name = args["model"]

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
yolomodel = {"config_path": "YOLO/yolov3.cfg",
             "model_weights_path": "YOLO/yolov3.weights",
             "coco_names": "YOLO/coco.names",
             "confidence_threshold": 0.5,
             "threshold": 0.3
             }

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt_path"],
                               conf["model_path"])
# net = cv2.dnn.readNetFromDarknet(yolomodel["config_path"], yolomodel["model_weights_path"])

np.random.seed(12345)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(layer_names)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
vs = cv2.VideoCapture(video_path)
# time.sleep(2.0)

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
H = None
W = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=conf["max_disappear"],
                     maxDistance=conf["max_distance"])
trackers = []
trackableObjects = {}

# keep the count of total number of frames
totalFrames = 0

# initialize the list of various points used to calculate the avg of
# the vehicle speed
points = [("A", "B", "C")]

# start the frames per second throughput estimator
fps = FPS().start()

addHeader(csv_name)

firstFrameVisited = False

# loop over the frames of the stream
while True:
    # grab the next frame from the stream, store the current
    # timestamp, and store the new date

    ret, frame = vs.read()

    if not firstFrameVisited:
        processFirstFrame(frame)
        firstFrameVisited = True

    ts = datetime.now()
    newDate = ts.strftime("%m-%d-%y")
    # check if the frame is None, if so, break out of the loop
    if frame is None:
        break

    # resize the frame
    if conf["yellowbox_casernes"]:
        findYellowBoxCasernes(frame)
    else:
        findYellowBoxDeschartes(frame)

    frame = imutils.resize(frame, width=conf["frame_width"])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        meterPerPixel = conf["distance"] / W

    # initialize our list of bounding box rectangles returned by
    # either (1) our object detector or (2) the correlation trackers
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % conf["track_object"] == 0:
        # initialize our new set of object trackers
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
        net.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5, 127.5, 127.5])
        # blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        # net.setInput(blob)
        detections = net.forward()
        # detections = np.array(detections)

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if confidence > conf["confidence"]:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a car, ignore it
                if CLASSES[idx] != "car" and CLASSES[idx] != "bus":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing
    # throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, if there is a trackable object and its speed has
        # not yet been estimated then estimate it
        elif not to.estimated:
            # check if the direction of the object has been set, if
            # not, calculate it, and set it
            # if to.direction is None:
            #	y = [c[1] for c in to.centroids]
            #	direction = centroid[1] - np.mean(y)
            #	to.direction = direction

            if conf["yellowbox_casernes"]:
                # getting x-coor and y-coor of left end of yellow box
                A_xCoor = conf["yellowbox_zone_casernes"]["A_xCoor"]
                A_yCoor = conf["yellowbox_zone_casernes"]["A_yCoor"]
                B_xCoor = conf["yellowbox_zone_casernes"]["B_xCoor"]
                B_yCoor = conf["yellowbox_zone_casernes"]["B_yCoor"]
                C_xCoor = conf["yellowbox_zone_casernes"]["C_xCoor"]
                C_yCoor = conf["yellowbox_zone_casernes"]["C_yCoor"]
            else:
                A_xCoor = conf["yellowbox_zone_deschartes"]["A_xCoor"]
                B_xCoor = conf["yellowbox_zone_deschartes"]["B_xCoor"]
                C_xCoor = conf["yellowbox_zone_deschartes"]["C_xCoor"]
                D_xCoor = conf["yellowbox_zone_deschartes"]["D_xCoor"]
                E_xCoor = conf["yellowbox_zone_deschartes"]["E_xCoor"]
                F_xCoor = conf["yellowbox_zone_deschartes"]["F_xCoor"]
                G_xCoor = conf["yellowbox_zone_deschartes"]["G_xCoor"]
                H_xCoor = conf["yellowbox_zone_deschartes"]["H_xCoor"]
                I_xCoor = conf["yellowbox_zone_deschartes"]["I_xCoor"]

            # check to see if timestamp has been noted for
            # point A
            if to.timestamp["A"] == 0:
                # if the centroid's x-coordinate is greater than
                # the corresponding point then set the timestamp
                # as current timestamp and set the position as the
                # centroid's x-coordinate and centroid[1] > B_yCoor and centroid[1] < A_yCoor
                if (centroid[0] > A_xCoor):
                    to.timestamp["A"] = (ts)
                    # print('Entry point')
                    # print(centroid[1])
                    to.position["A"] = centroid[0]
                    to.firstPoint = True

            # check to see if timestamp has been noted for
            # point B
            if to.timestamp["B"] == 0:
                # if the centroid's x-coordinate is greater than
                # the corresponding point then set the timestamp
                # as current timestamp and set the position as the
                # centroid's x-coordinate
                if (centroid[0] > B_xCoor):
                    to.timestamp["B"] = ts
                    # print('Midpoint')
                    # print(centroid[1])
                    to.midPoint = True
                    to.position["B"] = centroid[0]

            # check to see if timestamp has been noted for
            # point C
            if to.timestamp["C"] == 0:
                # if the centroid's x-coordinate is greater than
                # the corresponding point then set the timestamp
                # as current timestamp and set the position as the
                # centroid's x-coordinate
                if (centroid[0] > C_xCoor):
                    to.timestamp["C"] = ts
                    # print('Exit point')
                    # print(centroid[1])
                    to.position["C"] = centroid[0]
                    to.lastPoint = True

            # check to see if the vehicle is past the last point and
            # the vehicle's speed has not yet been estimated, if yes,
            # then calculate the vehicle speed and log it if it's
            # over the limit
            if ((to.firstPoint and to.midPoint and not to.estimated and not centroid[0] > 700) or (
                    to.midPoint and to.lastPoint and not to.estimated and not centroid[0] > 700)):
                # initialize the list of estimated speeds
                estimatedSpeeds = []

                # loop over all the pairs of points and estimate the
                # vehicle speed
                for (i, j, k) in points:
                    # calculate the distance in pixels
                    # d = to.position[j] - to.position[i]
                    # distanceInPixels = abs(d)

                    # check if the distance in pixels is zero, if so,
                    # skip this iteration
                    # if distanceInPixels == 0:
                    # continue

                    # calculate the time in hours
                    timeAtEntry = to.timestamp[i]
                    timeAtMiddle = to.timestamp[j]
                    timeAtExit = to.timestamp[k]
                    if timeAtEntry is not 0 and timeAtMiddle is not 0 and timeAtExit is not 0:
                        t1 = to.timestamp[j] - to.timestamp[i]
                        t2 = to.timestamp[k] - to.timestamp[j]
                    elif timeAtEntry is 0 and timeAtMiddle is not 0 and timeAtExit is not 0:
                        t1 = 0
                        t2 = to.timestamp[k] - to.timestamp[j]
                    elif timeAtEntry is not 0 and timeAtMiddle is not 0 and timeAtExit is 0:
                        t1 = to.timestamp[j] - to.timestamp[i]
                        t2 = 0
                    else:
                        t1 = 0
                        t2 = 0
                    if t1 is 0:
                        timeInSeconds = abs(t2.total_seconds())
                    elif t2 is 0:
                        timeInSeconds = abs(t1.total_seconds())
                    else:
                        timeInSeconds = (abs(t1.total_seconds()) + abs(t2.total_seconds())) / 2
                    print(objectID)
                    print(centroid[1])
                    print(timeInSeconds)
                    print()
                    timeInHours = timeInSeconds / (60 * 60)

                # if math.isnan(timeInSeconds):
                #	print("Error in recording time")
                # else:
                #	print("Saving data")
                #	saveInCsv(objectID,timeInSeconds)

                if conf['test_model']:
                    testModel(objectID, timeInSeconds, model_name)

                # set the object as estimated
                to.estimated = True
            # print("[INFO] Speed of the vehicle that just passed"\
            #	" is: {:.2f} MPH".format(to.speedMPH))

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
                    , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,
                   (0, 255, 0), -1)

    # if the *display* flag is set, then display the current frame
    # to the screen and record if a user presses a key addHeader()
    if conf["display"]:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# close any open windows
cv2.destroyAllWindows()

# clean up
print("[INFO] cleaning up...")
# vs.stop()
