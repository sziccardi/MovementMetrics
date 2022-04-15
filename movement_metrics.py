from asyncio import gather
from fileinput import filename
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from enum import Enum

video_fps = 30
video_pix_per_m = -1
vel_blocks = 15

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

plot_type_dict = dict({"point cloud":PlotType.POINT_CLOUD, "centroid of motion":PlotType.CENTROID, 
    "position spectrum":PlotType.POS_SPEC, "distance from center":PlotType.CENT_DIST, 
    "normalized distance from center":PlotType.CENT_DIST_NORM, "velocity heat map":PlotType.VEL_HEAT_MAP, 
    "speed over time":PlotType.TOTAL_VEL_OVER_TIME, "velocity over time":PlotType.VEL_OVER_TIME, 
    "aperature over time":PlotType.APERATURE, "accelerometer tree":PlotType.ACCEL_TREE})
keypoint_names = ['head','spine_top','shoulder_right','elbow_right','wrist_right','shoulder_left', 'elbow_left','wrist_left','spine_base','hip_right','knee_right','ankle_right','hip_left', 'knee_left','ankle_left','eye_right','eye_left','ear_right','ear_left','big_toe_left', 'little_toe_left','heel_left','big_toe_right','little_toe_right','heel_right','palm_left', 'thumb_base_left','thumb_1_left','thumb_2_left','thumb_tip_left','pointer_base_left', 'pointer_1_left','pointer_2_left','pointer_tip_left','middle_base_left','middle_1_left', 'middle_2_left','middle_tip_left','ring_base_left','ring_1_left','ring_2_left','ring_tip_left', 'pinky_base_left','pinky_1_left','pinky_2_left','pinky_tip_left','palm_right','thumb_base_right', 'thumb_1_right','thumb_2_right','thumb_tip_right','pointer_base_right','pointer_1_right', 'pointer_2_right','pointer_tip_right','middle_base_right','middle_1_right','middle_2_right', 'middle_tip_right','ring_base_right','ring_1_right','ring_2_right','ring_tip_right', 'pinky_base_right','pinky_1_right','pinky_2_right','pinky_tip_right']
keypoint_nums = list(np.arange(1,len(keypoint_names)+1))
stoi_map = dict(zip(keypoint_names, keypoint_nums))
itos_map = dict(zip(keypoint_nums, keypoint_names))

def ReadDataFromList(files):
    vals =[]
    for filename in files:
        #print(filename)
        if ".json" in filename:
            with open(filename, 'r', encoding='utf-8') as f: 
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

def ReadDataInFolder(file_path):
    vals =[]
    for filename in os.listdir(file_path):
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

# [frame, x/y/c, keypoint]
def PlotPointCloud(data, keypoints):
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        plt.xlabel("x Position (m)")
        plt.ylabel("y Position (m)")
        plt.xlim([0, 1280 / video_pix_per_m])
        plt.ylim([0, 720 / video_pix_per_m])
    else:
        plt.xlabel("x Pixel Position")
        plt.ylabel("y Pixel Position")
        plt.xlim([0, 1280])
        plt.ylim([0, 720])
    for i,point in enumerate(keypoints):
        start_iter = (point - 1)
        
        np_vals = np.array(data) 
        
        z_x = np.abs(stats.zscore(np_vals[:,0,start_iter]))
        z_y = np.abs(stats.zscore(np_vals[:,1,start_iter]))
        select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
        vals_cleaned = np_vals[select,:,start_iter]
        
        vals_cleaned[:,0] /= scale
        vals_cleaned[:,1] /= scale
        plt.scatter(vals_cleaned[:,0], vals_cleaned[:,1], s=1)
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]
    mid_x /= scale
    mid_y /= scale
    plt.scatter(mid_x, mid_y, s=1, color="black")
    labels = [itos_map[x] for x in keypoints]
    labels.append("center")
    plt.legend(labels, markerscale=6)
    plt.title("Position Point Cloud")
    figure = plt.gcf()
    figure.set_size_inches(5.25, 3.5)

def PlotCentroid(data, keypoints):
    np_vals = np.array(data)
    selected_vals = []
    
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]

    for point in keypoints:
        start_iter = (point - 1)
        np_vals = np.array(data) 
        
        selected = np_vals[:,:,start_iter]
        selected_vals.append(selected)
    
    selected_vals = np.stack(selected_vals, axis=2)
    np_means_avg = np.mean(selected_vals, axis=2)
    
    z_x = np.abs(stats.zscore(np_means_avg[:,0]))
    z_y = np.abs(stats.zscore(np_means_avg[:,1]))
    select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
    vals_cleaned = np_vals[select,:,start_iter]


    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        plt.xlabel("x Position (m)")
        plt.ylabel("y Position (m)")
    else:
        plt.xlabel("x Pixel Position")
        plt.ylabel("y Pixel Position")
    shifted_x = [(x - mid_x) / scale for x in vals_cleaned[:,0]]
    shifted_y = [(y - mid_y) / scale for y in vals_cleaned[:,1]]
    
    plt.scatter(shifted_x, shifted_y, s=1)
    
    total_mean_x = np.mean(shifted_x)
    total_mean_y = np.mean(shifted_y)
    plt.plot([total_mean_x], [total_mean_y], marker="o", markersize=3, markeredgecolor='black', markerfacecolor="black")

    plt.title("Centroid of movement")
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    figure = plt.gcf()
    figure.set_size_inches(5.25, 3.5)

def PlotLocSpectrum(data, keypoints):
    x_dict = {}
    y_dict = {}
    fig, ax = plt.subplots(2)
    fig.tight_layout(pad=2.0)
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        ax[0].set_title('Spectrum of X Values (m)')
        ax[1].set_title('Spectrum of Y Values (m)')
    else:
        ax[0].set_title('Spectrum of X Values (pixels)')
        ax[1].set_title('Spectrum of Y Values (pixels)')
    for point in keypoints:
        start_iter = (point - 1)
        np_vals = np.array(data)

        z_x = np.abs(stats.zscore(np_vals[:,0,start_iter]))
        z_y = np.abs(stats.zscore(np_vals[:,1,start_iter]))
        select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
        vals_cleaned = np_vals[select,:,start_iter]
       
        name = itos_map[point]

        x_dict[name] = [(1280 - x) / scale for x in vals_cleaned[:,0]]
        y_dict[name] = vals_cleaned[:,1] / scale
    
    int_arg = stoi_map['spine_top'] - 1
    z_x = np.abs(stats.zscore(np_vals[:,0,int_arg]))
    z_y = np.abs(stats.zscore(np_vals[:,1,int_arg]))
    select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
    vals_cleaned = np_vals[select,:,int_arg]
    mid_x = [(1280 - x) / scale for x in vals_cleaned[:,0]]
    mid_y = vals_cleaned[:,1] / scale
    x_dict["center"] = mid_x
    y_dict["center"] = mid_y

    x_vals = [x for x in x_dict.values()]
    x_keys = [x for x in x_dict.keys()]
    ax[0].boxplot(x_vals[::-1], vert=False)
    ax[1].boxplot(y_dict.values())
    ax[0].set_yticklabels(x_keys[::-1])
    ax[1].set_xticklabels(y_dict.keys())
    fig.set_size_inches(5.25, 5.5)

def PlotDistFromCenter(normalized, data, keypoints):
    np_vals = np.array(data)
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        plt.xlabel("x Distance (m)")
        plt.ylabel("y Distance (m)")
    else:
        plt.xlabel("x Pixel Distance")
        plt.ylabel("y Pixel Distance")
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]
    labels = []
    for point in keypoints:
        if (point != stoi_map['spine_top']):
            start_iter = (point - 1)
            
            selected = selected = np_vals[:,:,start_iter]
            if normalized:
                selected[:,0] = abs(mid_x - selected[:,0]) / scale
                selected[:,1] = abs(mid_y - selected[:,1]) / scale
            else:    
                selected[:,0] = (selected[:,0] - mid_x) / scale
                selected[:,1] = (selected[:,1] - mid_y) / scale

            
            z_x = np.abs(stats.zscore(selected[:,0]))
            z_y = np.abs(stats.zscore(selected[:,1]))
            select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
            vals_cleaned = selected[select,:]

            plt.scatter(vals_cleaned[:,0], vals_cleaned[:,1], s=1)
            labels.append(itos_map[point])
    
    plt.legend(labels, markerscale=6)
    if normalized:
        plt.title("Normalized Distance from Center")
    else:
        plt.title("Distance from Center")
    
    if not normalized:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    figure = plt.gcf()
    figure.set_size_inches(5.25, 3.5)


def PlotVelocityHeatMap(data, keypoints):
    np_vals = np.array(data)
    heat_map = np.zeros((720, 1280))
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
    title = ""
    for i, point in enumerate(keypoints):
        start_iter = (point - 1)
        
        z_x = np.abs(stats.zscore(np_vals[:,0,start_iter]))
        z_y = np.abs(stats.zscore(np_vals[:,1,start_iter]))
        select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
        vals_cleaned = np_vals[select,:,start_iter]

        xs = vals_cleaned[:,0, start_iter]
        xs /= vel_blocks
        xs = np.where(xs>(1279), int(1279), xs)
        ys = vals_cleaned[:,1, start_iter]
        ys /= vel_blocks
        ys = np.where(ys>(719), int(719), ys)

        #confs = np_vals[:,2, start_iter]
        x_ints = np.full_like(xs, 0) # x positions that have velocity
        np.rint(xs, out=x_ints)
        x_ints = x_ints[:-1]
        
        y_ints = np.full_like(ys, 0) # y positions that have velocity
        np.rint(ys, out=y_ints)
        y_ints = y_ints[:-1]

        cp_np_vals = np_vals[:,:, start_iter].copy()
        shifted_np_vals = cp_np_vals[1:,:]
        new_np_vals = np_vals[:-1,:, start_iter]

        delta_pos = shifted_np_vals - new_np_vals
        vels = delta_pos * float(video_fps)
        
        total_vels = [np.sqrt(vels[i,0] * vels[i, 0] + vels[i, 1] * vels[i,1]) for i in range(vels.shape[0])]

        heat_map[y_ints.astype(int), x_ints.astype(int)] += total_vels
        title += (str(itos_map[point]))
        if i < len(keypoints) - 1:
            title += " and "
        
    im = plt.imshow(heat_map, cmap='cool', vmin = 0.0, vmax = 250.0)
    
    plt.title("Velocity of " + title)
    plt.xlabel("x Pixel Position")
    plt.ylabel("y Pixel Position")
    plt.xlim([0, 1280/ vel_blocks])
    plt.ylim([0, 720 / vel_blocks])
    plt.colorbar(im)

def PlotSpeedOverTime(data, keypoints):
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        plt.ylabel("total speed (m/s)")
    else:
        plt.ylabel("total speed (pixels/s)")
    
    for point in keypoints:
        start_iter = (point - 1)
        
        np_vals = np.array(data)
        name = itos_map[point]
        
        z_x = np.abs(stats.zscore(np_vals[:,0,start_iter]))
        z_y = np.abs(stats.zscore(np_vals[:,1,start_iter]))
        select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
        vals_cleaned = np_vals[select,:,start_iter]

        xs = vals_cleaned[:,0]
        xs = np.where(xs>1279, 1279, xs)
        ys = vals_cleaned[:,1]
        ys = np.where(ys>719, 719, ys)

        confs = vals_cleaned[:,2]
        x_ints = np.full_like(xs, 0) # x positions that have velocity
        np.rint(xs, out=x_ints)
        x_ints = x_ints[:-1]
        
        y_ints = np.full_like(ys, 0) # y positions that have velocity
        np.rint(ys, out=y_ints)
        y_ints = y_ints[:-1]

        cp_np_vals = vals_cleaned[:,:].copy()
        shifted_np_vals = cp_np_vals[1:,:]
        new_np_vals = vals_cleaned[:-1,:]

        delta_pos = shifted_np_vals - new_np_vals
        vels = delta_pos / float(scale) * float(video_fps)
        
        total_vels = [np.sqrt(vels[i,0] * vels[i, 0] + vels[i, 1] * vels[i,1]) for i in range(vels.shape[0])]

        # avged_vels = []
        # running_total = 0
        # for i, val in enumerate(total_vels):
        #     running_total += val
        #     if i % vel_blocks == 0:
        #         avg_vel = running_total / vel_blocks
        #         avged_vels.append(avg_vel)
        #         running_total = 0
        avged_vels = np.convolve(total_vels, np.ones(vel_blocks), 'valid') / vel_blocks


        plt.plot(np.arange(0, len(total_vels) / float(video_fps), 1.0 / float(video_fps))[:len(avged_vels)], avged_vels)

    
    labels = [itos_map[x] for x in keypoints]
    labels.append("center")
    plt.legend(labels, markerscale=6)
    plt.title("Speed over time")
    plt.xlabel("seconds")
    figure = plt.gcf()
    figure.set_size_inches(5.25, 3.5)
    

def PlotVelocitiesOverTime(data, keypoints):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        ax1.set_ylabel("velocity (m/s)")
        ax2.set_ylabel("velocity (m/s)")
    else:
        ax1.set_ylabel("velocity (pixels/s)")
        ax2.set_ylabel("velocity (pixels/s)")
    
    my_min = 100000000000
    my_max = -100000000000
    for point in keypoints:
        start_iter = (point - 1)
        
        np_vals = np.array(data)
        name = itos_map[point]

        xs = np_vals[:,0, start_iter]
        xs = np.where(xs>1279, 1279, xs)
        ys = np_vals[:,1, start_iter]
        ys = np.where(ys>719, 719, ys)
        cs = np_vals[:,2,start_iter]
        # mask = cs > 0.6
        # xs = xs[mask]
        # ys = ys[mask]

        cp_np_vals = np_vals[:,:, start_iter].copy()
        shifted_np_vals = cp_np_vals[1:,:]
        new_np_vals = np_vals[:-1,:, start_iter]

        delta_pos = shifted_np_vals - new_np_vals
        vels = delta_pos / float(scale) * float(video_fps)
        list_vels_x = vels[:,0].tolist()
        list_vels_y = vels[:,1].tolist()

        # avged_vels_x = []
        # avged_vels_y = []
        # running_total_x = 0
        # running_total_y = 0
        # for i, x_val in enumerate(list_vels_x):
        #     y_val = list_vels_y[i]
        #     running_total_x += x_val
        #     running_total_y += y_val
        #     if i % vel_blocks == 0:
        #         avg_vel_y = running_total_y / vel_blocks
        #         avg_vel_x = running_total_x / vel_blocks
        #         avged_vels_x.append(avg_vel_x)
        #         avged_vels_y.append(avg_vel_y)
        #         running_total_x = 0
        #         running_total_y = 0

        avged_vels_x = np.convolve(list_vels_x, np.ones(vel_blocks), 'valid') / vel_blocks
        avged_vels_y = np.convolve(list_vels_y, np.ones(vel_blocks), 'valid') / vel_blocks
        temp_max = np.amax(avged_vels_y)
        temp_min = np.amin(avged_vels_y)
        if (temp_max > my_max):
            my_max = temp_max
        if (temp_min < my_min):
            my_min = temp_min

        my_x_x = np.arange(0, len(list_vels_x)/float(video_fps), 1.0/float(video_fps))[:len(avged_vels_x)]
        my_x_y = avged_vels_x
        z_x_x = np.abs(stats.zscore(my_x_x))
        z_x_y = np.abs(stats.zscore(my_x_y))
        select_x = [(a < 3) and (b < 3) for a, b in zip(z_x_x, z_x_y)]

        my_y_x = np.arange(0, len(list_vels_y)/float(video_fps), 1.0/float(video_fps))[:len(avged_vels_y)]
        my_y_y = avged_vels_y
        z_y_x = np.abs(stats.zscore(my_x_x))
        z_y_y = np.abs(stats.zscore(my_x_y))
        select_y = [(a < 3) and (b < 3) for a, b in zip(z_y_x, z_y_y)]
        
        ax1.plot(my_x_x[select_x], my_x_y[select_x])
        ax2.plot(my_y_x[select_y], my_y_y[select_y])

    ax1.set_ylim([my_min, my_max])
    ax2.set_ylim([my_min, my_max])
    labels = [itos_map[x] for x in keypoints]
    labels.append("center")
    plt.legend(labels, markerscale=6)
    ax1.set_title("X velocity over time")
    ax2.set_title("Y velocity over time")
    plt.xlabel("seconds")
    
    fig.set_size_inches(5.25, 5.5)

    
def PlotAccelerometerTree(data, keypoints):
    
    np_vals = np.array(data)
    
    heat_map = np.zeros((720, 1280))
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
    title = ""
    max_accels = []
    for i, point in enumerate(keypoints):
        start_iter = (point - 1)
        
        z_x = np.abs(stats.zscore(np_vals[:,0,start_iter]))
        z_y = np.abs(stats.zscore(np_vals[:,1,start_iter]))
        select = [(a < 3) and (b < 3) for a, b in zip(z_x, z_y)]
        vals_cleaned = np_vals[select,:,start_iter]

        cp_np_vals = vals_cleaned[:,:, start_iter].copy()
        shifted_np_vals = cp_np_vals[1:,:]
        new_np_vals = vals_cleaned[:-1,:, start_iter]

        delta_pos = shifted_np_vals - new_np_vals
        vels = delta_pos * float(video_fps)
        
        cp_np_vels = vels.copy()
        shifted_np_vels = cp_np_vels[1:,:]
        new_np_vels = vels[:-1,:]
        delta_vel = shifted_np_vels - new_np_vels
        accels = delta_vel * float(video_fps)
        
        total_accels = [np.sqrt(accels[i,0] * accels[i, 0] + accels[i, 1] * accels[i,1]) for i in range(accels.shape[0])]

        max_accels.append([])
        for j in range(int(len(total_accels) / vel_blocks) - 1):
            max_val = max(total_accels[j * vel_blocks:(j+1)*vel_blocks])
            max_accels[i].append(max_val)

        # heat_map[y_ints.astype(int), x_ints.astype(int)] += total_vels
        title += (str(itos_map[point]))
        if i < len(keypoints) - 1:
            title += " and "
    
    
    # natural log of non-dominant/dominant
    x_val = np.array([ np.log(max_accels[0][i] / max_accels[1][i]) for i in range(len(max_accels[0]))])
    y_val = np.array([ max_accels[0][i] + max_accels[1][i] for i in range(len(max_accels[0]))])
    y_val = y_val / np.min(y_val)

    plt.hist2d(x_val, y_val, vel_blocks)



def PlotAperatureOverTime(data, keypoints):
    if len(keypoints) < 3:
        print("WARNING: you need at least 3 points to compute an aperature.")
        return
    
    total_area = np.zeros(len(data))
    for i in range(len(keypoints) - 2):
        key1 = int(keypoints[0])
        key2 = int(keypoints[i+1])
        key3 = int(keypoints[i+2])
        np_vals = np.array(data)
        x1 = np_vals[:, 0, key1]
        x2 = np_vals[:, 0, key2]
        x3 = np_vals[:, 0, key3]
        y1 = np_vals[:, 1, key1]
        y2 = np_vals[:, 1, key2]
        y3 = np_vals[:, 1, key3]
        area_of_tri = np.abs((x1*y2 + x2*y3 + x3*y1 - y1*x2 - y2*x3 - y3*x1) / 2.0)
        total_area += area_of_tri
    
    avged_vels = np.convolve(total_area, np.ones(vel_blocks), 'valid') / vel_blocks
    plt.plot(np.arange(0, len(avged_vels)/ video_fps, 1.0/video_fps)[:len(avged_vels)], avged_vels)

    plt.title("Aperature over time")
    plt.xlabel("time (seconds)")
    plt.ylabel("aperature (pixels)")
    figure = plt.gcf()
    figure.set_size_inches(5.25, 3.5)
    
    


def Plot(data, keypoints, type, filename = ""):
    
    if type == PlotType.POINT_CLOUD:
        PlotPointCloud(data, keypoints)
    elif type == PlotType.CENTROID:
        PlotCentroid(data, keypoints)
    elif type == PlotType.POS_SPEC:
        PlotLocSpectrum(data, keypoints)
    elif type == PlotType.CENT_DIST:
        PlotDistFromCenter(False, data, keypoints)
    elif type == PlotType.CENT_DIST_NORM:
        PlotDistFromCenter(True, data, keypoints)
    elif type == PlotType.VEL_HEAT_MAP:
        PlotVelocityHeatMap(data, keypoints)
    elif type == PlotType.TOTAL_VEL_OVER_TIME:
        PlotSpeedOverTime(data, keypoints)
    elif type == PlotType.VEL_OVER_TIME:
        PlotVelocitiesOverTime(data, keypoints)
    elif type == PlotType.APERATURE:
        PlotAperatureOverTime(data, keypoints)
    elif type == PlotType.ACCEL_TREE:
        PlotAccelerometerTree(data, keypoints)
    

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    return True

def run_script(frame_files, plot_type, keypoints, fps, pix_in_m, cov_width):
    fig_numbers = [x.num for x in plt._pylab_helpers.Gcf.get_all_fig_managers()]
    if len(fig_numbers) > 0:
        plt.figure().clear()
        plt.close('all')
        plt.cla()
        plt.clf()
    video_fps = fps
    video_pix_per_m = pix_in_m
    vel_blocks = cov_width

    real_keypoints = [stoi_map[k] for k in keypoints]

    data = ReadDataFromList(frame_files)
    flag = Plot(data, real_keypoints, plot_type_dict[plot_type], "TEMP.png")
    if not flag:
        print("ERROR: Couldn't find that plot")

if __name__ == '__main__':
    fig_numbers = [x.num for x in plt._pylab_helpers.Gcf.get_all_fig_managers()]
    if len(fig_numbers) > 0:
        plt.figure().clear()
        plt.close('all')
        plt.cla()
        plt.clf()
    file_path = ''
    plot_type = PlotType.NONE
    keypoints = []

    gather_keypoints = False
    for i, arg in enumerate(sys.argv):
        if gather_keypoints:
            if arg in stoi_map:
                int_arg = stoi_map[arg]
                keypoints.append(int_arg)

        if arg == '--frame_files':
            gather_keypoints = False
            file_path = sys.argv[i+1]
        if arg == '--point_cloud':
            gather_keypoints = True
            plot_type = PlotType.POINT_CLOUD
        if arg == '--centroid':
            gather_keypoints = True
            plot_type = PlotType.CENTROID
        if arg == '--pos_spectrum':
            gather_keypoints = True
            plot_type = PlotType.POS_SPEC
        if arg == '--dist_from_center':
            gather_keypoints = True
            plot_type = PlotType.CENT_DIST
        if arg == '--dist_from_center_norm':
            gather_keypoints = True
            plot_type = PlotType.CENT_DIST_NORM
        if arg == '--velocity_heat_map':
            gather_keypoints = True
            plot_type = PlotType.VEL_HEAT_MAP
        if arg == '--speed_over_time':
            gather_keypoints = True
            plot_type = PlotType.TOTAL_VEL_OVER_TIME
        if arg == '--velocity_over_time':
            gather_keypoints = True
            plot_type = PlotType.VEL_OVER_TIME
        if arg == '--accelerometer_tree':
            gather_keypoints = True
            plot_type = PlotType.ACCEL_TREE

        # if arg == '--aperture_over_time':
        #     gather_keypoints = True
        #     plot_type = PlotType.APERATURE

        if arg == '--fps':
            gather_keypoints = False
            video_fps = sys.argv[i+1]
        if arg == '--pix_to_m':
            gather_keypoints = False
            pix = float(sys.argv[i+1])
            m = float(sys.argv[i+2])
            video_pix_per_m = pix / m
        if arg == '--avg_width':
            gather_keypoints = False
            vel_blocks = int(sys.argv[i+1])


    data = ReadDataInFolder(file_path)
    flag = Plot(data, keypoints, plot_type)
    if not flag:
        print("ERROR: Couldn't find that plot")

    
    
    
    