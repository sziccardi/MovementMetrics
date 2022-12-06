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

    REL_ANG_HIST = 3
    REL_ANG_OVER_TIME = 4


plot_type_dict = dict({"relative position":PlotType.REL_POS, "relative position over time":PlotType.REL_POS_OVER_TIME, "relative angle histogram":PlotType.REL_ANG_HIST, "relative angle over time":PlotType.REL_ANG_OVER_TIME})
keypoint_names = ['nose', 'left mouth', 'right mouth',  'left shoulder', 'right shoulder',  'left elbow', 'right elbow','left wrist', 'right wrist', 'left pinky', 'right pinky', 'left index finger', 'right index finger', 'left thumb', 'right thumb']
keypoint_nums = list(np.arange(0,len(keypoint_names)))
stoi_map = dict(zip(keypoint_names, keypoint_nums))
itos_map = dict(zip(keypoint_nums, keypoint_names))

# ANGLE globals
bin_w = 10.0
extend_lim = 165
contract_lim = 30

# ALL globals
conf_filter = 0.8


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
        if ind < len(data) - 1: 
            if (start_less and data[ind] < data[ind+1]) or (not start_less and data[ind] > data[ind+1]):
                k = ind+1
                while k < len(data)-1 and np.sign(data[k]-lim) == np.sign(data[k+1]-lim):
                    temp_total_count = temp_total_count+1
                    k=k+1
                temp_num_count = temp_num_count+1
    return temp_total_count, temp_num_count


#Plotting
# data[frame, x/y/z/v, keypoint]
def GetRelativePositionData(data, keypoints):
    print("GetRelativePositionData")
    data_list = list(data.values())
    np_vals = np.array(data_list)
        
    left_should = stoi_map['left shoulder']
    right_should = stoi_map['right shoulder']
    
    mid_x = (np_vals[:,0,left_should] + np_vals[:,0,right_should]) / 2.0
    mid_y = (np_vals[:,1,left_should] + np_vals[:,1,right_should]) / 2.0
    mid_z = (np_vals[:,2,left_should] + np_vals[:,2,right_should]) / 2.0
    labels = []
    processed_data = []
    key_list = list(data.keys())
    for start_iter in keypoints:
            
        selected = np_vals[:,:,start_iter]
        selected[:,0] = (selected[:,0] - mid_x)
        selected[:,1] = (selected[:,1] - mid_y)
        stacked = np.column_stack((selected[:,0], selected[:,1], selected[:,3]))
        print(stacked.shape)
        dict_processed_data = {key_list[i]: stacked[i] for i in range(len(key_list))}

        processed_data.append(dict_processed_data)
        
        labels.append(itos_map[start_iter])

    
    return processed_data, labels

def GetRelativePositionOverTimeData(data, keypoints, fps, vel_blocks):
    print("GetRelativePositionOverTimeData")
    data_list = list(data.values())
    np_vals = np.array(data_list)
    
    left_should = stoi_map['left shoulder']
    right_should = stoi_map['right shoulder']
    mid_x = (np_vals[:,0,left_should] + np_vals[:,0,right_should]) / 2.0
    mid_y = (np_vals[:,1,left_should] + np_vals[:,1,right_should]) / 2.0

    labels = []
    processed_data_h = []
    processed_data_v = []
    for start_iter in keypoints:
        
        selected = np_vals[:,:,start_iter]
        selected[:,0] = (selected[:,0] - mid_x)
        selected[:,1] = (selected[:,1] - mid_y)

        rel_pos = selected[:,:3]
        ts = np.arange(0, len(rel_pos) / float(fps), 1.0 / float(fps))
        my_scale = min(len(ts), len(rel_pos))
        avged_pos_x = np.convolve(selected[:my_scale,0], np.ones(vel_blocks), 'valid') / vel_blocks
        avged_pos_y = np.convolve(selected[:my_scale,1], np.ones(vel_blocks), 'valid') / vel_blocks
        
        horiz = np.column_stack((ts[:len(avged_pos_x)], avged_pos_x, selected[:,3]))
        print("horiz shape", horiz.shape)
        key_list = list(data.keys())
        dict_horiz = {key_list[i]: horiz[i] for i in range(len(key_list))}
        vert = np.column_stack((ts[:len(avged_pos_y)], avged_pos_y, selected[:,3]))
        dict_vert = {key_list[i]: vert[i] for i in range(len(key_list))}
        processed_data_h.append(dict_horiz)
        processed_data_v.append(dict_vert)

        
        labels.append(itos_map[start_iter])
    
    return [processed_data_h, processed_data_v], labels

def GetAngleOverTimeData(data, keypoints, video_fps, vel_blocks):
    axes_labels = []
    axes_labels.append("time (s)")
    axes_labels.append("angle (deg)")
    
    processed_data = []
    labels = []

    print("GetAngleOverTimeData")
    data_list = list(data.values())
    np_vals = np.array(data_list)
    key_list = list(data.keys())
    for start_iter in keypoints:
        
        p1_i = None 
        p3_i = None
        p2_i = start_iter
        
        if start_iter == (stoi_map['left shoulder']):
            p1_i = stoi_map['right shoulder']
            p3_i = stoi_map['left elbow']
        elif start_iter == (stoi_map['left elbow']):
            p1_i = stoi_map['left shoulder']
            p3_i = stoi_map['left wrist']
        elif start_iter == (stoi_map['left wrist']):
            p1_i = stoi_map['left elbow']
            p3_i = stoi_map['left index finger']
        elif start_iter == (stoi_map['right shoulder']):
            p1_i = stoi_map['left shoulder']
            p3_i = stoi_map['right elbow']
        elif start_iter == (stoi_map['right elbow']):
            p1_i = stoi_map['right shoulder']
            p3_i = stoi_map['right wrist']
        elif start_iter == (stoi_map['right wrist']):
            p1_i = stoi_map['right elbow']
            p3_i = stoi_map['right index finger']

        if p1_i and p2_i and p3_i:
            a_vec = np_vals[:,:,p2_i] - np_vals[:,:,p1_i]
            b_vec = np_vals[:,:,p3_i] - np_vals[:,:,p2_i]

            a_len = np.sqrt(np.sum(np.multiply(a_vec,a_vec), axis=1))
            b_len = np.sqrt(np.sum(np.multiply(b_vec,b_vec), axis=1))

            a_dot_b = np.sum(np.multiply(a_vec,b_vec), axis=1)

            angle = np.arccos(a_dot_b / (a_len * b_len))
            angle = np.degrees(angle)

            avged_ang = np.convolve(angle, np.ones(vel_blocks), 'valid') / vel_blocks
            confs = np_vals[:,3,p2_i] + np_vals[:,3,p1_i] + np_vals[:,3,p3_i]
            confs = confs / 3


            angs = (np.column_stack((np.arange(0, len(avged_ang) / float(video_fps), 1.0 / float(video_fps))[:len(avged_ang)], avged_ang, confs)))
            dict_angs = {key_list[i]: angs[i] for i in range(len(key_list))}
            processed_data.append(dict_angs)

            labels.append(itos_map[start_iter])

    return [processed_data], labels, axes_labels

def GetAngleHistogramData(data, keypoints, video_fps, vel_blocks):
    processed_data, labels, axes_labels = GetAngleOverTimeData(data, keypoints, video_fps, vel_blocks)

    axes_labels = ["angle (deg)","frequency (frames)"]

    actual_data = []
    for start_iter in range(len(keypoints)):
        
        avged_ang = np.array(list(processed_data[0][start_iter].values()))
        confs = avged_ang[:,2] > conf_filter
        avged_ang = avged_ang[confs,1]
        indicies = (avged_ang / bin_w)
        indicies = np.floor(indicies)
        temp = [0 for i in range(int(180.0 / bin_w))]
        for i in range(int(180.0 / bin_w)):
            temp[i] = np.sum(indicies == i)
        actual_data.append(temp)
    return actual_data, labels, axes_labels
    
def GetPlotSpecificInfo(plot_type):
    if plot_type_dict[plot_type] == PlotType.REL_POS_OVER_TIME:
        return [conf_filter]
    if plot_type_dict[plot_type] == PlotType.REL_POS:
        return [conf_filter]
    if plot_type_dict[plot_type] == PlotType.REL_ANG_HIST:
        return [conf_filter, bin_w]
    if plot_type_dict[plot_type] == PlotType.REL_ANG_OVER_TIME:
        return [conf_filter, extend_lim, contract_lim]

def run_script_get_data(data, img_size, plot_type, keypoints, fps, cov_width):
    real_keypoints = [stoi_map[k] for k in keypoints]
    
    print("running ", plot_type)
    processed_data = data_labels = ax_labels = None
    if plot_type_dict[plot_type] == PlotType.REL_POS:
        processed_data, data_labels = GetRelativePositionData(data, real_keypoints)
    elif plot_type_dict[plot_type] == PlotType.REL_POS_OVER_TIME:
        processed_data, data_labels = GetRelativePositionOverTimeData(data, real_keypoints, fps, cov_width)
    elif plot_type_dict[plot_type] == PlotType.REL_ANG_HIST:
        processed_data, data_labels, ax_labels = GetAngleHistogramData(data, real_keypoints, fps, cov_width)
    elif plot_type_dict[plot_type] == PlotType.REL_ANG_OVER_TIME:
        processed_data, data_labels, ax_labels = GetAngleOverTimeData(data, real_keypoints, fps, cov_width)


    if processed_data == None or data_labels == None:
        print("WARNING: Could not process data as provided.")
    
    
    return processed_data, data_labels

    
    
    
    