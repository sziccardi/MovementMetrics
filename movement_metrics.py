from statistics import mean
import json
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from enum import Enum
import math

class PlotType(Enum):
    NONE = 0

    REL_POS = 1
    REL_POS_OVER_TIME = 2

plot_type_dict = dict({"relative position":PlotType.REL_POS, "relative position over time":PlotType.REL_POS_OVER_TIME})
keypoint_names = ['head','spine_top','shoulder_right','elbow_right','wrist_right','shoulder_left', 'elbow_left','wrist_left']
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
    vals = {}
    x = []
    y = []
    c = []
    for filename in files:
        if ".json" in filename:
            with open(filename, 'r', encoding='utf-8') as f: 
                #print("read " + filename)
                lines = f.readlines()
                json_file = json.loads(lines[0])
                if len(json_file['people']) > 0:
                    # when multiple people are in the frame, only read the main one
                    person_i = 0
                    dist_from_cent = 2000
                    max_keypoints = 0

                    data_array = json_file['people'][person_i]['pose_keypoints_2d']
                    x_pose = data_array[::3]
                    y_pose = [700 - x for x in data_array[1::3]]
                    c_pose = data_array[2::3]

                    for person in range(len(json_file['people'])):
                        filter = np.array(c_pose) < conf_filter
                        new_num = np.sum(filter)
                        if min(8,new_num) > max_keypoints:
                            max_keypoints = min(8,new_num)
                            person_i = person
                        else:
                            x_dif = img_size[0]/2.0 - json_file['people'][person]['pose_keypoints_2d'][0]
                            y_dif = img_size[1]/2.0 - json_file['people'][person]['pose_keypoints_2d'][1]
                            new_len = math.sqrt(x_dif*x_dif + y_dif*y_dif)
                            if new_len < dist_from_cent:
                                dist_from_cent = new_len
                                person_i = person
                    
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

                end_val = filename.find("_keypoints.json")
                start_val = filename[:end_val].rfind("_")
                n = int(filename[start_val+1:end_val])
                vals[n] = [x,y,c]

    print("read in ", len(vals))
    return vals


#Metric helpers
def getMin(data):
    return min(data)

def getMax(data):
    return max(data)

def getMean(data):
    return mean(data)

def getVariance(data):
    return np.var(data)

def getCoVariance(data1, data2):
    return np.cov(data1, data2)

def getSTD(data):
    return np.std(data)


def getPeaks(data, my_threshold):
    return signal.find_peaks(data, threshold=my_threshold)


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
def GetRelativePositionData(data, keypoints):
    data_list = list(data.values())
    np_vals = np.array(data_list)
        
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]
    labels = []
    processed_data = []
    for point in keypoints:
        start_iter = (point - 1)
            
        selected = np_vals[:,:,start_iter]
        selected[:,0] = (selected[:,0] - mid_x)
        selected[:,1] = (selected[:,1] - mid_y)
        
        key_list = list(data.keys())
        dict_processed_data = {key_list[i]: selected[i] for i in range(len(key_list))}

        processed_data.append(dict_processed_data)
        
        labels.append(itos_map[point])

    
    return processed_data, labels

def GetRelativePositionOverTimeData(data, keypoints, fps, vel_blocks):
    data_list = list(data.values())
    np_vals = np.array(data_list)
        
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]
    labels = []
    processed_data = []
    for point in keypoints:
        start_iter = (point - 1)
        
        selected = np_vals[:,:,start_iter]
        selected[:,0] = (selected[:,0] - mid_x)
        selected[:,1] = (selected[:,1] - mid_y)

        rel_pos = selected[:,:2]
        ts = np.arange(0, len(rel_pos) / float(fps), 1.0 / float(fps))
        my_scale = min(len(ts), len(rel_pos))
        temp = []
        avged_pos_x = np.convolve(selected[:my_scale,0], np.ones(vel_blocks), 'valid') / vel_blocks
        avged_pos_y = np.convolve(selected[:my_scale,1], np.ones(vel_blocks), 'valid') / vel_blocks
        
        horiz = np.column_stack((ts[:len(avged_pos_x)], avged_pos_x, selected[:,2]))
        key_list = list(data.keys())
        dict_horiz = {key_list[i]: horiz[i] for i in range(len(key_list))}
        vert = np.column_stack((ts[:len(avged_pos_y)], avged_pos_y, selected[:,2]))
        dict_vert = {key_list[i]: vert[i] for i in range(len(key_list))}
        temp.append(dict_horiz)
        temp.append(dict_vert)

        processed_data.append(temp)
        
        labels.append(itos_map[point])
    
    return processed_data, labels


def GetPlotSpecificInfo(plot_type):
    if plot_type_dict[plot_type] == PlotType.REL_POS_OVER_TIME:
        return [conf_filter]
    if plot_type_dict[plot_type] == PlotType.REL_POS:
        return [conf_filter]

def run_script_get_data(frame_files, img_size, plot_type, keypoints, fps, cov_width):
    real_keypoints = [stoi_map[k] for k in keypoints]
    data = ReadDataFromList(frame_files, img_size)
    print("running ", plot_type)
    processed_data = data_labels = ax_labels = None
    if plot_type_dict[plot_type] == PlotType.REL_POS:
        processed_data, data_labels = GetRelativePositionData(data, real_keypoints)
    elif plot_type_dict[plot_type] == PlotType.REL_POS_OVER_TIME:
        processed_data, data_labels = GetRelativePositionOverTimeData(data, real_keypoints, fps, cov_width)
    
    if processed_data == None or data_labels == None:
        print("WARNING: Could not process data as provided.")
    
    return processed_data, data_labels

    
    
    
    