import cv2
import mediapipe as mp
import time
import math
import csv
import pandas as pd
from os import walk
from pathlib import Path


class poseDetector():
    def __init__(self, 
                static_image_mode=False,
                model_complexity = 2,
                smooth_landmarks = True,
                enable_segmentation = False,
                smooth_segmentation = False,
                min_detection_confidence = 0.5,
                min_tracking_confidence = 0.5):
        #self.mode = mode
        #self.upBody = upBody
        #self.smooth = smooth
        #self.detectionCon = detectionCon
        #self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = static_image_mode,
                                     model_complexity = model_complexity,
                                     smooth_landmarks = smooth_landmarks,
                                     enable_segmentation = enable_segmentation,
                                     smooth_segmentation = smooth_segmentation,
                                     min_detection_confidence = min_detection_confidence,
                                     min_tracking_confidence = min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([lm.x, lm.y, lm.z, lm.visibility])
                if draw:
                   cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        annotated_image = img.copy()
        if draw:
            self.mpDraw.draw_landmarks(
                annotated_image,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS,
                landmark_drawing_spec= mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        return self.lmList,annotated_image
import os.path
def main(path,filename,draw = True):
    if 'Aashirvad' not in filename and  'Aditya' not in filename and 'Toshith' not in filename:
        return
    base_path = path+'input_videos/'+filename
    if  os.path.isfile(base_path+'.mp4'):
        file_path = path+'input_videos/'+filename+'.mp4'
    elif os.path.isfile(base_path+'.MOV') :
        file_path = path+'input_videos/'+filename+'.MOV'
    elif os.path.isfile(base_path+'.mov'):
        file_path = path+'input_videos/'+filename+'.mov'
    else:
        print(base_path, "continue")
        return
    print("input file :",file_path)

    cap = cv2.VideoCapture(file_path)
    p = time.time()
    detector = poseDetector()
    
    a_file = Path(path+'csv_vis/'+filename+'.csv')
    a_file.parent.mkdir(exist_ok=True, parents=True)
    a_file = open(path+'csv_vis/'+filename+'.csv', "w+")
    a_file.close()

    # label the data for better understanding
    labels = ['frame']
    cord = {}
    cord[0] = 'x'
    cord[1] = 'y'
    cord[2] = 'z'
    cord[3] = 'visibility'
    for i in range(33*4):
        t = i // 4
        c = i % 4
        labels.append(cord[c]+'_'+str(t))
    df = pd.DataFrame(columns = labels)
    df.set_index('frame', drop=True, append=False, inplace=False, verify_integrity=False)
    i = 0
    frame = 0

    if draw:
        _,img = cap.read()
        h, w, c = img.shape

    # choose codec according to format needed
    if draw:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("video saved at {}".format(path+'output_videos/'+filename+'.mp4'))
        # video = cv2.VideoWriter(path+'output_videos/'+filename+'.mp4', fourcc, 24, (w, h))

    while True:
        success, img = cap.read()
        if(not success):
            break
        try:
            img = detector.findPose(img,draw)
            lmList,annotated_image = detector.findPosition(img, draw=draw)

            normalized_vectors = [frame]
            for pt in lmList:
                normalized_vectors.extend([pt[0], pt[1], pt[2], pt[3]])
            df.loc[i] = normalized_vectors
            i += 1
        except:
            print("error in frame {}",format(frame))
        if draw:
            cv2.imshow("Image", annotated_image)
            cv2.waitKey(1)
        frame += 1

    df.to_csv(path+'csv_vis/'+filename+'.csv', index=False)

    if draw:
        cv2.destroyAllWindows()
        # video.release()

    return



if __name__ == "__main__":
    #print("This script's main function required to be called from another script.")
    #path = "C:/Users/gcgro/Documents/Thesis/fitbuddy/Demo/"

    # get file name of all files in folder input_videos/
    # f = []
    # for (root, dirs, filenames) in walk('input_videos/'):
    #     for dir in dirs:
    #         for (_, _, filenames) in walk(os.path.join(root, dir)):
    #             for filename in filenames:
    #                 f.append(os.path.join(dir, filename))
    f = []
    for (root, dirs, filenames) in walk('input_videos/'):
        for dir in dirs:
            # dir='Plank'
            for (_, _, filenames) in walk(os.path.join(root, dir)):
                for filename in filenames:
                    f.append(os.path.join(dir, filename))
        # break
    print(f)

    for fname in f:
        print(fname)
        main("",fname[:-4],draw=False)