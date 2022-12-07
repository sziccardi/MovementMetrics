import numpy as np
import sys
import datetime
import os
import cv2
import csv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def process_img(image, idx, solution, frame_loc):
    # Convert the BGR image to RGB before processing.
    results = solution.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks or not results.left_hand_landmarks or not results.right_hand_landmarks:
        print("missing landmarks in frame #", idx)
        cv2.imwrite(frame_loc+'/video_frames/frame_'+ str(idx).zfill(7) + '.png', image)
        return

    frame_points = []

    #  0 9 10 11 12 13 14    
    # NOSE MOUTH_LEFT MOUTH_RIGHT LEFT_SHOULDER RIGHT_SHOULDER LEFT_ELBOW RIGHT_ELBOW
    #x and y are normalized to [0.0, 1.0] by the image width and height respectively.
    #z represents the landmark depth with the depth at center of the head being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
    px = results.pose_world_landmarks.landmark[0].x
    py = results.pose_world_landmarks.landmark[0].y
    pz = results.pose_world_landmarks.landmark[0].z
    pv = results.pose_world_landmarks.landmark[0].visibility
    frame_points.append(px)
    frame_points.append(py)
    frame_points.append(pz)
    frame_points.append(pv)
    for i in range(9,15):
        px = results.pose_landmarks.landmark[i].x
        py = results.pose_landmarks.landmark[i].y
        pz = results.pose_landmarks.landmark[i].z
        pv = results.pose_landmarks.landmark[i].visibility
        frame_points.append(px)
        frame_points.append(py)
        frame_points.append(pz)
        frame_points.append(pv)
    
    # x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
    # z represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
    
    for i in range(len(results.left_hand_landmarks.landmark)):
        px = results.left_hand_landmarks.landmark[i].x
        py = results.left_hand_landmarks.landmark[i].y
        pz = results.left_hand_landmarks.landmark[i].z
        pv = results.left_hand_landmarks.landmark[i].visibility
        frame_points.append(px)
        frame_points.append(py)
        frame_points.append(pz)
        frame_points.append(pv)
    
    for i in range(len(results.right_hand_landmarks.landmark)):
        px = results.right_hand_landmarks.landmark[i].x
        py = results.right_hand_landmarks.landmark[i].y
        pz = results.right_hand_landmarks.landmark[i].z
        pv = results.right_hand_landmarks.landmark[i].visibility
        frame_points.append(px)
        frame_points.append(py)
        frame_points.append(pz)
        frame_points.append(pv)
    
    annotated_image = image.copy()
    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        annotated_image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(
        annotated_image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(frame_loc+'/video_frames/frame_'+ str(idx).zfill(7) + '.png', annotated_image)

    return frame_points

if __name__ == '__main__':
    images_loc = None
    video_file = None
    file_loc = None
    for i, arg in enumerate(sys.argv):
        if arg == '--images_loc':
            images_loc = sys.argv[i+1]
        if arg == '--video_file':
            video_file = sys.argv[i+1]
        if arg == '--save_loc':
            file_loc = sys.argv[i+1]
    
    st = datetime.datetime.now()
    keypoints = ["nose", "mouth left", "mouth right", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "left thumb base", "left thumb 1", "left thumb 2", "left thumb 3", "left index base", "left index 1", "left index 2", "left index 3", "left middle base", "left middle 1", "left middle 2", "left middle 3", "left ring base", "left ring 1", "left ring 2", "left ring 3", "left pinky base", "left pinky 1", "left pinky 2", "left pinky 3", "right wrist", "right thumb base", "right thumb 1", "right thumb 2", "right thumb 3", "right index base", "right index 1", "right index 2", "right index 3", "right middle base", "right middle 1", "right middle 2", "right middle 3", "right ring base", "right ring 1", "right ring 2", "right ring 3", "right pinky base", "right pinky 1", "right pinky 2", "right pinky 3"]

    header = []
    for key in keypoints:
        header.append(key + " x")
        header.append(key + " y")
        header.append(key + " z")
        header.append(key + " v")

    with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
    
        if images_loc is not None and video_file is None:
            print("STARTING STATIC IMAGE PROCESSING")
            # For static images:
            save_loc = images_loc
            if file_loc is not None:
                save_loc = file_loc
            with open(save_loc+'/processed/pose_info.csv', 'w', newline='\n') as csvfile:
                posewriter = csv.writer(csvfile, delimiter=',')
                posewriter.writerow(header)
                for idx, file in enumerate(os.listdir(images_loc)):
                    if ".png" in file:
                        image = cv2.imread(images_loc+"/"+file)
                        image_height, image_width, _ = image.shape

                        poses = process_img(image, idx, holistic, save_loc+'/processed')
                        if poses is not None:
                            posewriter.writerow(poses)
                        
        elif video_file is not None:
            print("STARTING VIDEO FILE PROCESSING")
            start = video_file.rfind("/")
            save_loc = video_file[:start]
            if file_loc is not None:
                save_loc = file_loc
            #For video file input:
            with open(save_loc+'/processed/pose_info.csv', 'w', newline='\n') as csvfile:
                posewriter = csv.writer(csvfile, delimiter=',')
                posewriter.writerow(header)

                cap = cv2.VideoCapture(video_file)
                
                idx = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        break

                    poses = process_img(image, idx, holistic, save_loc+'/processed')
                    if poses is not None:
                        posewriter.writerow(poses)
                    
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    
                    idx = idx + 1
                cap.release()
                print("Processed ", idx, " frames")
        elif file_loc is not None:
            #For webcam input:
            with open(images_loc+'/processed/pose_info.csv', 'w', newline='\n') as csvfile:
                posewriter = csv.writer(csvfile, delimiter=',')
                posewriter.writerow(header)

                cap = cv2.VideoCapture(0)
                idx = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue
                    poses = process_img(image, idx, holistic, file_loc+'/processed')
                    if poses is not None:
                        posewriter.writerow(poses)
                    
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    cap.release()

                    idx = idx + 1
                print("Processed ", idx, " frames")
        else:
            print("WARNING: No valid input actions found.")
    
    print("Started at ", st)
    ct = datetime.datetime.now()
    print("Ended at ", ct)
    print("Elapsed time: ", ct - st)