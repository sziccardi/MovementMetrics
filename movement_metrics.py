import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class PlotType(Enum):
    NONE = 0
    POINT_CLOUD = 1

def ReadData(file_path):
    vals =[]
    for filename in os.listdir(file_path):
        with open(os.path.join(file_path, filename), 'r') as f: 
            print("read " + filename)
            lines = f.readlines()
            json_file = json.loads(lines[0])

            data_array = json_file['people'][0]['pose_keypoints_2d']
            x_pose = data_array[::3]
            y_pose = [700 - x for x in data_array[1::3]]
            c_pose = data_array[2::3]

            data_array = json_file['people'][0]['hand_right_keypoints_2d']
            x_right_hand = data_array[::3]
            y_right_hand = [700 - x for x in data_array[1::3]]
            c_right_hand = data_array[2::3]

            data_array = json_file['people'][0]['hand_left_keypoints_2d']
            x_left_hand = data_array[::3]
            y_left_hand = [700 - x for x in data_array[1::3]]
            c_left_hand = data_array[2::3]

            x = []
            x.extend(x_pose)
            x.extend(x_right_hand)
            x.extend(x_left_hand)
            y = []
            y.extend(y_pose)
            y.extend(y_right_hand)
            y.extend(y_left_hand)
            c = []
            c.extend(c_pose)
            c.extend(c_right_hand)
            c.extend(c_left_hand)

            vals.append([x,y,c])
    return vals

# [frame, x/y/c, keypoint]
def PlotPointCloud(data, keypoints):
    for point in keypoints:
        start_iter = (point - 1)
        
        np_vals = np.array(data)
        plt.scatter(np_vals[:,0, start_iter], np_vals[:,1, start_iter], alpha=np_vals[:,2, start_iter], s=1)
    
    labels = [itos_map[x] for x in keypoints]
    plt.legend(labels)
    plt.title("Position Point Cloud")
    plt.xlabel("x Pixel Position")
    plt.ylabel("y Pixel Position")
    plt.xlim([0, 1280])
    plt.ylim([0, 720])
    plt.show()

def Plot(data, keypoints, type):
    match type:
        case PlotType.POINT_CLOUD:
            PlotPointCloud(data, keypoints)
            return True
        case _:
            return False


if __name__ == '__main__':
    keypoint_names = ['head','spine_top','shoulder_left','elbow_left','wrist_left','shoulder_right',
    'elbow_right','wrist_right','spine_base','hip_left','knee_left','ankle_left','hip_right',
    'knee_right','ankle_right','eye_left','eye_right','ear_left','ear_right','big_toe_right',
    'little_toe_right','heel_right','big_toe_left','little_toe_left','heel_left','palm_left',
    'thumb_base_left','thumb_1_left','thumb_2_left','thumb_tip_left','pointer_base_left',
    'pointer_1_left','pointer_2_left','pointer_tip_left','middle_base_left','middle_1_left',
    'middle_2_left','middle_tip_left','ring_base_left','ring_1_left','ring_2_left','ring_tip_left',
    'pinky_base_left','pinky_1_left','pinky_2_left','pinky_tip_left','palm_right','thumb_base_right',
    'thumb_1_right','thumb_2_right','thumb_tip_right','pointer_base_right','pointer_1_right',
    'pointer_2_right','pointer_tip_right','middle_base_right','middle_1_right','middle_2_right',
    'middle_tip_right','ring_base_right','ring_1_right','ring_2_right','ring_tip_right',
    'pinky_base_right','pinky_1_right','pinky_2_right','pinky_tip_right']
    keypoint_nums = list(np.arange(1,len(keypoint_names)+1))
    stoi_map = dict(zip(keypoint_names, keypoint_nums))
    itos_map = dict(zip(keypoint_nums, keypoint_names))

    file_path = ''
    plot_type = PlotType.NONE
    keypoints = []

    gather_keypoints = False
    for i, arg in enumerate(sys.argv):
        if gather_keypoints:
            print("arg = " + arg)
            int_arg = stoi_map[arg]
            print(int_arg)
            keypoints.append(int_arg)

        if arg == '--frame_files':
            gather_keypoints = False
            file_path = sys.argv[i+1]
        if arg == '--point_cloud':
            gather_keypoints = True
            plot_type = PlotType.POINT_CLOUD


    data = ReadData(file_path)
    flag = Plot(data, keypoints, plot_type)

    
    
    
    