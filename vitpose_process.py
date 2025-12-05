import sys
import math
import os
import cv2
import csv
import argparse
import matplotlib
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    required=True,
    help='path to the input video'
)

parser.add_argument(
    '--output-loc',
    dest='output_loc',
    required=True,
    help='path to save location of the video and csv'
)

parser.add_argument(
    '--save-name',
    dest='save_name',
    help='name of the output video. if blank, no video output'
)

parser.add_argument(
    '--save-velocity',
    dest='use_vel',
    action='store_true',
    help='flag for whether or not to save velocity in csv'
)


parser.add_argument(
    '--det-conf',
    dest='det_conf',
    default=0.3,
    type=float,
    help='detection confidence threshold'
)
parser.add_argument(
    '--pose-model',
    dest='pose_model',
    choices=[
        'usyd-community/vitpose-base',
        'usyd-community/vitpose-base-simple',
        'usyd-community/vitpose-base-coco-aic-mpii',
        'usyd-community/vitpose-plus-small',
        'usyd-community/vitpose-plus-base',
        'usyd-community/vitpose-plus-large',
        'usyd-community/vitpose-plus-huge'
    ],
    default='usyd-community/vitpose-base'
)
args = parser.parse_args()



# Load ViTPose.
print(f"Pose Model: {args.pose_model}")
image_processor = AutoProcessor.from_pretrained(args.pose_model, use_fast=True)
model = VitPoseForPoseEstimation.from_pretrained(args.pose_model, device_map='cpu')


def process_img(image, person_boxes):
    """
    :param image: Image in PIL image format.
    :param person_bboxes: Batched person boxes in [[x, y, w, h], ...] format.
    """
    pose_time_start = time.time()
    inputs = image_processor(
        image, boxes=[person_boxes], return_tensors='pt'
    ).to('cpu')
    
    dataset_index = torch.tensor([0], device='cpu') # must be a tensor of shape (batch_size,)
    if len(person_boxes) != 0:
        if 'plus' in args.pose_model:
            with torch.no_grad():
                outputs = model(**inputs, dataset_index=dataset_index)
        else:
            with torch.no_grad():
                outputs = model(**inputs)
        
        pose_results = image_processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes]
        )
    pose_time_end = time.time()
    pose_fps = 1 / (pose_time_end-pose_time_start)
    if len(person_boxes) == 0:
        return [], pose_fps
    image_pose_result = pose_results[0]
    
    return image_pose_result, pose_fps

def draw_keypoints(outputs, image):
    """
    :param outputs: Outputs from the keypoint detector.
    :param image: Image in PIL Image format.
    Returns:
        image: Annotated image Numpy array format.
    """
    edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6), (11, 12)
    ]

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # the `outputs` is list which in-turn contains the dictionaries 
    for i, pose_result in enumerate(outputs):
        keypoints = pose_result['keypoints'].cpu().detach().numpy()
        # proceed to draw the lines if the confidence score is above 0.9
        keypoints = keypoints[:, :].reshape(-1, 2)
        for p in range(keypoints.shape[0]):
            # draw the keypoints
            cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                        3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            # uncomment the following lines if you want to put keypoint number
            # cv2.putText(image, f'{p}', (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for ie, e in enumerate(edges):
            # get different colors for the edges
            rgb = matplotlib.colors.hsv_to_rgb([
                ie/float(len(edges)), 1.0, 1.0
            ])
            rgb = rgb*255
            # join the keypoint pairs to draw the skeletal structure
            cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                    (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                    tuple(rgb), 2, lineType=cv2.LINE_AA)
    return image

# 0 - nose
# 1 - left eye
# 2 - right eye
# 3 - left ear
# 4 - right ear
# 5 - left shoulder
# 6 - right shoulder
# 7 - left elbow
# 8 - right elbow
# 9 - left wrist
# 10 - right wrist
# 11 - left hip
# 12 - right hip
# 13 - left knee
# 14 - right knee
# 15 - left ankle
# 16 - right ankle
header = ['nose_x', 'nose_y', 
    'left_eye_x', 'left_eye_y', 
    'right_eye_x', 'right_eye_y',
    'left_ear_x', 'left_ear_y', 
    'right_ear_x', 'right_ear_y',
    'left_shoulder_x', 'left_shoulder_y', 
    'right_shoulder_x', 'right_shoulder_y',
    'left_elbow_x', 'left_elbow_y', 
    'right_elbow_x', 'right_elbow_y',
    'left_wrist_x', 'left_wrist_y', 
    'right_wrist_x', 'right_wrist_y',
    'left_hip_x', 'left_hip_y', 
    'right_hip_x', 'right_hip_y',
    'left_knee_x', 'left_knee_y', 
    'right_knee_x', 'right_knee_y',
    'left_ankle_x', 'left_ankle_y', 
    'right_ankle_x', 'right_ankle_y'
    ]

header_wvel = ['nose_x', 'nose_y', 'nose_vx', 'nose_vy', 'nose_vt', 
    'left_eye_x', 'left_eye_y', 'left_eye_vx', 'left_eye_vy', 'left_eye_vt', 
    'right_eye_x', 'right_eye_y', 'right_eye_vx','right_eye_vy', 'right_eye_vt',  
    'left_ear_x', 'left_ear_y','left_ear_vx', 'left_ear_vy','left_ear_vt', 
    'right_ear_x', 'right_ear_y','right_ear_vx', 'right_ear_vy','right_ear_vt',
    'left_shoulder_x', 'left_shoulder_y','left_shoulder_vx', 'left_shoulder_vy','left_shoulder_vy', 
    'right_shoulder_x', 'right_shoulder_y','right_shoulder_vx', 'right_shoulder_vy','right_shoulder_vt',
    'left_elbow_x', 'left_elbow_y', 'left_elbow_vx', 'left_elbow_vy', 'left_elbow_vt',
    'right_elbow_x', 'right_elbow_y','right_elbow_vx', 'right_elbow_vy','right_elbow_vt',
    'left_wrist_x', 'left_wrist_y','left_wrist_vx', 'left_wrist_vy','left_wrist_vt', 
    'right_wrist_x', 'right_wrist_y','right_wrist_vx', 'right_wrist_vy','right_wrist_vt',
    'left_hip_x', 'left_hip_y', 'left_hip_vx', 'left_hip_vy','left_hip_vt', 
    'right_hip_x', 'right_hip_y','right_hip_vx', 'right_hip_vy','right_hip_vt',
    'left_knee_x', 'left_knee_y','left_knee_vx', 'left_knee_vy','left_knee_vt', 
    'right_knee_x', 'right_knee_y','right_knee_vx', 'right_knee_vy','right_knee_vt',
    'left_ankle_x', 'left_ankle_y','left_ankle_vx', 'left_ankle_vy','left_ankle_vt', 
    'right_ankle_x', 'right_ankle_y', 'right_ankle_vx', 'right_ankle_vy', 'right_ankle_vt'
    ]



def run_video(my_video_loc, my_output_loc, withVel = False):
    cap = cv2.VideoCapture(my_video_loc)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_fps = int(cap.get(5))

    save_name = args.save_name
    # if args.save_name == None:
    #     j = my_video_loc.rfind('.')
    #     i = my_video_loc.rfind('\\')
    #     if i < 0:
    #         i = my_video_loc.rfind('/')

    #     save_name = args.input[i+1:j]
    out = None
    if args.save_name != None:
        print(f"Writing video to {my_output_loc}/{save_name}.mp4")
        out = cv2.VideoWriter(
            f'{my_output_loc}/{save_name}.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            video_fps, 
            (frame_width, frame_height)
        )

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.
    error_count = 0
    with open(f'{my_output_loc}/pose_info.csv', 'w', newline='\n') as csvfile:
        posewriter = csv.writer(csvfile, delimiter=',')
        if withVel:
            posewriter.writerow(header_wvel)
        else:
            posewriter.writerow(header)
        prevs = None
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                start_time = time.time()
                #bboxes, det_fps = detect_objects(image=image)
                bboxes = [[0,0,frame_width, frame_height]]
                frame_points = []
                image_pose_result, pose_fps = process_img(image=image, person_boxes=bboxes)
                if withVel:
                    person_pose = image_pose_result[0]
                    for i, (keypoint, label, score) in enumerate(zip( person_pose["keypoints"], person_pose["labels"], person_pose["scores"])):
                        x, y = keypoint
                        
                        frame_points.append(x.cpu().detach().item())
                        frame_points.append(y.cpu().detach().item())
                        if prevs is None:
                            frame_points.append(None)
                            frame_points.append(None)
                            frame_points.append(None)
                        else:
                            prev_px, prev_py = prevs[i]

                            dx = (x.cpu().detach().item() - prev_px.cpu().detach().item())
                            dy = (y.cpu().detach().item() - prev_py.cpu().detach().item())

                            dp = math.sqrt(dx*dx + dy*dy)
                            frame_points.append(dx * video_fps)
                            frame_points.append(dy * video_fps)
                            frame_points.append(dp * video_fps)

                    prevs = person_pose["keypoints"]
                else:
                    person_pose = image_pose_result[0]
                    for keypoint, label, score in zip( person_pose["keypoints"], person_pose["labels"], person_pose["scores"]):
                        x, y = keypoint
                        frame_points.append(x.cpu().detach().item())
                        frame_points.append(y.cpu().detach().item())

                try:
                    posewriter.writerow(frame_points)
                except:
                    posewriter.writerow([])
                    print("WARNING: somethin went wrong when saving frame points at frame #", frame_count)

                if out is not None:
                    result = draw_keypoints(image_pose_result, image)
                end_time = time.time()
                forward_pass_time = end_time - start_time
                    
                # Get the current fps.
                fps = 1 / (forward_pass_time)
                # Add `fps` to `total_fps`.
                total_fps += fps
                # if frame_count % 60 == 0:
                #     print(fps)
                # Increment frame count.
                frame_count += 1
                if out is not None:
                    cv2.putText(
                        result,
                        f"FPS: {fps:0.1f} | Pose FPS: {pose_fps:0.1f} ",
                        (15, 25),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(0, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    out.write(result)
                #cv2.imshow('Prediction', result)
                # Press `q` to exit
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break
            else:
                print("WARNING: Couldn't load frame ", frame_count)
                error_count += 1
            if error_count > 3:
                break
        # Release VideoCapture().
        cap.release()
        # Close all frames and video windows.
        cv2.destroyAllWindows()
        avg_fps = 0
        try:
            avg_fps = total_fps/frame_count
        except:
            print("WARNING: Frame count 0")
        print(f"Average FPS for {save_name}: {avg_fps}")


def run_videos(my_file_loc, my_output_loc, withVel = False):
    for idx in tqdm(range(len(os.listdir(my_file_loc)))):
        files = os.listdir(my_file_loc)
        file = files[idx]
        i = file.rfind('.')
        filename = file[:i]
        print("Processing video: ", file)
        
        if not os.path.exists(my_output_loc+ '/' + filename):
            print("HAD TO MAKE A FOLDER")
            os.makedirs(my_output_loc+ '/' + filename)
            run_video(my_file_loc+ '/' + file, my_output_loc+ '/' + filename, withVel)
        else:
            print(filename, " has already been processed")



if __name__ == '__main__':
    if '.' in args.input:
        print("STARTING RUN VIDEO")
        run_video(args.input, args.output_loc,args.use_vel)
    else:
        print("STARTING TO RUN VIDEO DIR")
        run_videos(args.input, args.output_loc,args.use_vel)