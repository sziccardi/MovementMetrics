import os
from statistics import mean, median
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from enum import Enum
import math

class PlotType(Enum):
    NONE = 0
    POINT_CLOUD = 1
    CENTROID = 2
    POS_SPEC = 3
    CENT_DIST = 4
    CENT_DIST_NORM = 5

    VEL_HEAT_MAP = 6
    TOTAL_VEL_OVER_TIME = 7
    VEL_OVER_TIME = 8

    APERATURE = 9

    ACCEL_TREE = 10

    REL_POS = 11
    REL_POS_OVER_TIME = 12
    MOV_HEAT_MAP = 13
    ANGLE_OVER_TIME = 14
    ANGLE_HIST = 15
    
    REL_ANGLES = 16

plot_type_dict = dict({"point cloud":PlotType.POINT_CLOUD, "centroid of motion":PlotType.CENTROID, 
    "position spectrum":PlotType.POS_SPEC, "distance from center":PlotType.CENT_DIST, 
    "normalized distance from center":PlotType.CENT_DIST_NORM, "velocity heat map":PlotType.VEL_HEAT_MAP, 
    "speed over time":PlotType.TOTAL_VEL_OVER_TIME, "velocity over time":PlotType.VEL_OVER_TIME, 
    "aperature over time":PlotType.APERATURE, "accelerometer tree":PlotType.ACCEL_TREE,
    "relative position":PlotType.REL_POS, "relative position over time":PlotType.REL_POS_OVER_TIME, 
    "movement heatmap":PlotType.MOV_HEAT_MAP, "angles over time":PlotType.ANGLE_OVER_TIME, 
    "angle histogram":PlotType.ANGLE_HIST, "relative angles":PlotType.REL_ANGLES})
keypoint_names = ['head','spine_top','shoulder_right','elbow_right','wrist_right','shoulder_left', 'elbow_left','wrist_left']
#'spine_base','hip_right','knee_right','ankle_right','hip_left', 'knee_left','ankle_left','eye_right','eye_left','ear_right','ear_left','big_toe_left', 'little_toe_left','heel_left','big_toe_right','little_toe_right','heel_right','palm_left', 'thumb_base_left','thumb_1_left','thumb_2_left','thumb_tip_left','pointer_base_left', 'pointer_1_left','pointer_2_left','pointer_tip_left','middle_base_left','middle_1_left', 'middle_2_left','middle_tip_left','ring_base_left','ring_1_left','ring_2_left','ring_tip_left', 'pinky_base_left','pinky_1_left','pinky_2_left','pinky_tip_left','palm_right','thumb_base_right', 'thumb_1_right','thumb_2_right','thumb_tip_right','pointer_base_right','pointer_1_right', 'pointer_2_right','pointer_tip_right','middle_base_right','middle_1_right','middle_2_right', 'middle_tip_right','ring_base_right','ring_1_right','ring_2_right','ring_tip_right', 'pinky_base_right','pinky_1_right','pinky_2_right','pinky_tip_right'
keypoint_nums = list(np.arange(1,len(keypoint_names)+1))
stoi_map = dict(zip(keypoint_names, keypoint_nums))
itos_map = dict(zip(keypoint_nums, keypoint_names))

# HEATMAP globals
box_w = 25

# ANGLE HIST globals
bin_w = 10.0

# ALL globals
conf_filter = 0.5

def ReadDataFromList(files, img_size):
    vals =[]
    x = []
    y = []
    c = []
    for filename in files:
        #print(filename)
        if ".json" in filename:
            with open(filename, 'r', encoding='utf-8') as f: 
                #print("read " + filename)
                lines = f.readlines()
                json_file = json.loads(lines[0])
                if len(json_file['people']) > 0:
                    # when multiple people are in the frame, only read the main one
                    person_i = 0
                    dist_from_cent = 0
                    for person in range(len(json_file['people'])):
                        # x distance of the right shoulder and the left shoulder
                        # in theory the person in frame focus has the largest shoulder width
                        x_dif = img_size[0]/2.0 - json_file['people'][person]['pose_keypoints_2d'][0]
                        y_dif = img_size[1]/2.0 - json_file['people'][person]['pose_keypoints_2d'][1]
                        if len(json_file['people'][person]['pose_keypoints_2d']) > 1:
                            x_dif = img_size[0]/2.0 - json_file['people'][person]['pose_keypoints_2d'][0]
                            y_dif = img_size[1]/2.0 - json_file['people'][person]['pose_keypoints_2d'][1]
                            new_len = math.sqrt(x_dif*x_dif + y_dif*y_dif)
                            if new_len < dist_from_cent:
                                dist_from_cent = new_len
                                person_i = person

                    data_array = json_file['people'][person_i]['pose_keypoints_2d']
                    x_pose = data_array[::3]
                    y_pose = [700 - x for x in data_array[1::3]]
                    c_pose = data_array[2::3]

                    #flipped left and right to respect participant left/right vs frame left/right
                    data_array = json_file['people'][person_i]['hand_left_keypoints_2d'] 
                    x_right_hand = data_array[::3]
                    y_right_hand = [700 - x for x in data_array[1::3]]
                    c_right_hand = data_array[2::3]

                    data_array = json_file['people'][person_i]['hand_right_keypoints_2d']
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
                else:
                    c = [0 for i in range(len(x))]

                vals.append([x,y,c])

    return vals

def ReadDataInFolder(file_path):
    vals =[]
    
    for filename in os.listdir(file_path+'/pose_info'):
        
        if ".json" in filename:
            with open(os.path.join(file_path, filename), 'r') as f: 
                #print("read " + filename)
                lines = f.readlines()
                json_file = json.loads(lines[0])

                data_array = json_file['people'][0]['pose_keypoints_2d']
                x_pose = data_array[::3]
                y_pose = [700 - x for x in data_array[1::3]]
                c_pose = data_array[2::3]

                #flipped left and right to respect participant left/right vs frame left/right
                data_array = json_file['people'][0]['hand_left_keypoints_2d'] 
                x_right_hand = data_array[::3]
                y_right_hand = [700 - x for x in data_array[1::3]]
                c_right_hand = data_array[2::3]

                data_array = json_file['people'][0]['hand_right_keypoints_2d']
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

#Metric helpers
def getMin(data):
    return min(data)

def getMax(data):
    return max(data)

def getMean(data):
    return mean(data)

def getMedian(data):
    return median(data)

def getQuartile(data, q):
    return np.percentile(data, q*25)

def getVariance(data):
    return np.var(data)

def getCoVariance(data1, data2):
    return np.cov(data1, data2)

def getSTD(data):
    return np.std(data)

def getSpread(data1, data2):
    c = np.cov(np.vstack((data1, data2)))
    w = np.linalg.eigvals(c)
    s = 5.991
    major_ax = np.sqrt(s * w[0])
    minor_ax = np.sqrt(s * w[1])

    return np.pi * major_ax * minor_ax

def getCorrelationR(data1, data2):
    return stats.pearsonr(data1, data2)

def getPeaks(data, my_threshold):
    return signal.find_peaks(data, threshold=my_threshold)

def getCorrelationCross(data1, data2):
    ccov = np.correlate(data1 - data1.mean(), data2 - data2.mean())
    ccov_test = np.correlate(data1 - data1.mean(), data2 - data2.mean(), mode='full')
    ccor = ccov / (len(data1) * data1.std() * data2.std())
    ccor_test = ccov_test / (len(data1) * data1.std() * data2.std())
    print(ccor)
    print(max(ccor_test))
    return ccor


def getAxesCrossedCounts(data, start_less):
    cross_indicies = np.where(np.diff(np.sign(data)))[0]
    temp_total_count = 0
    temp_num_count = 0
    for ind in cross_indicies:
        if (start_less and data[ind] < data[ind+1]) or (not start_less and data[ind] > data[ind+1]):
            k = ind+1
            while (len(data) > k+1) and (np.sign(data[k]) == np.sign(data[k+1])):
                temp_total_count = temp_total_count+1
                k=k+1
            temp_num_count = temp_num_count+1
    return temp_total_count, temp_num_count

def getValueCrossedCounts(data, less_than, lim):
    start_less = not less_than
    cross_indicies = np.where(data > lim)[0]
    if less_than:
        cross_indicies = np.where(data < lim)[0]
    temp_total_count = 0
    temp_num_count = 0
    for ind in cross_indicies:
        if ind < len(data) - 1 and (start_less and data[ind] < data[ind+1]) or (not start_less and data[ind] > data[ind+1]):
            k = ind+1
            while k < len(data)-1 and np.sign(data[k]-lim) == np.sign(data[k+1]-lim):
                temp_total_count = temp_total_count+1
                k=k+1
            temp_num_count = temp_num_count+1
    return temp_total_count, temp_num_count


#Plotting
# [frame, x/y/c, keypoint]
def GetRelativePositionData(data, keypoints, video_pix_per_m):
    np_vals = np.array(data)

    axes_labels = []
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        axes_labels.append("x Position (m)")
        axes_labels.append("y Position (m)")
    else:
        axes_labels.append("x Position (pixel)")
        axes_labels.append("y Position (pixel)")
        
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]
    labels = []
    processed_data = []
    for point in keypoints:
        start_iter = (point - 1)
            
        selected = np_vals[:,:,start_iter]
        selected[:,0] = (selected[:,0] - mid_x) / scale
        selected[:,1] = (selected[:,1] - mid_y) / scale
        
        z_x = np.abs(stats.zscore(selected[:,0]))
        z_y = np.abs(stats.zscore(selected[:,1]))
        select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
        vals_cleaned = selected[select,:]

        processed_data.append(vals_cleaned)
        
        labels.append(itos_map[point])
    
    return processed_data, labels, axes_labels

def GetRelativePositionOverTimeData(data, keypoints, fps, video_pix_per_m, vel_blocks):
    np_vals = np.array(data)

    axes_labels = []
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        axes_labels.append(["time (s)", "Horizontal position (m)"])
        axes_labels.append(["time (s)", "Vertical position (m)"])
    else:
        axes_labels.append(["time (s)", "Horizontal position (pixels)"])
        axes_labels.append(["time (s)", "Vertical position (pixels)"])
        
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]
    labels = []
    processed_data = []
    for point in keypoints:
        start_iter = (point - 1)
        
        selected = np_vals[:,:,start_iter]
        selected[:,0] = (selected[:,0] - mid_x) / scale
        selected[:,1] = (selected[:,1] - mid_y) / scale
        
        z_x = np.abs(stats.zscore(selected[:,0]))
        z_y = np.abs(stats.zscore(selected[:,1]))
        select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
        vals_cleaned = selected[select,:]

        rel_pos = vals_cleaned[:,:2]
        ts = np.arange(0, len(rel_pos) / float(fps), 1.0 / float(fps))
        my_scale = min(len(ts), len(rel_pos))
        temp = []
        avged_pos_x = np.convolve(vals_cleaned[:my_scale,0], np.ones(vel_blocks), 'valid') / vel_blocks
        avged_pos_y = np.convolve(vals_cleaned[:my_scale,1], np.ones(vel_blocks), 'valid') / vel_blocks
        temp.append(np.column_stack((ts[:len(avged_pos_x)], avged_pos_x, vals_cleaned[:,2])))
        temp.append(np.column_stack((ts[:len(avged_pos_y)], avged_pos_y, vals_cleaned[:,2])))
        processed_data.append(temp)

        #processed_data.append(np.column_stack((np.arange(0, len(rel_pos) / float(video_fps), 1.0 / float(video_fps))[:len(rel_pos)], rel_pos)))
        #processed_data.append(rel_pos)
        
        labels.append(itos_map[point])
    
    return processed_data, labels, axes_labels


def GetPlotSpecificInfo(plot_type):
    if plot_type_dict[plot_type] == PlotType.REL_POS_OVER_TIME:
        return [conf_filter]
    if plot_type_dict[plot_type] == PlotType.REL_POS:
        return [conf_filter]

def run_script_get_data(frame_files, img_size, plot_type, keypoints, fps, pix_in_m, cov_width):
    real_keypoints = [stoi_map[k] for k in keypoints]
    data = ReadDataFromList(frame_files, img_size)
    print("running ", plot_type)
    processed_data = data_labels = ax_labels = None
    if plot_type_dict[plot_type] == PlotType.REL_POS:
        processed_data, data_labels, ax_labels = GetRelativePositionData(data, real_keypoints, pix_in_m)
    elif plot_type_dict[plot_type] == PlotType.REL_POS_OVER_TIME:
        processed_data, data_labels, ax_labels = GetRelativePositionOverTimeData(data, real_keypoints, fps, pix_in_m, cov_width)
    
    if processed_data == None or data_labels == None:
        print("WARNING: Could not process data as provided.")
    
    return processed_data, data_labels, ax_labels

    
    
    
    