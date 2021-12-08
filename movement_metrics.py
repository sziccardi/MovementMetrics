import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

video_fps = 30
video_pix_per_m = -1

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


def ReadData(file_path):
    vals =[]
    for filename in os.listdir(file_path):
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
    for point in keypoints:
        start_iter = (point - 1)
        
        np_vals = np.array(data) 
        np_vals[:,0, start_iter] /= scale
        np_vals[:,1, start_iter] /= scale
        plt.scatter(np_vals[:,0, start_iter], np_vals[:,1, start_iter], alpha=np_vals[:,2, start_iter], s=1)
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]
    mid_x /= scale
    mid_y /= scale
    plt.scatter(mid_x, mid_y, s=1, color="black")
    labels = [itos_map[x] for x in keypoints]
    labels.append("center")
    plt.legend(labels)
    plt.title("Position Point Cloud")
    plt.show()

def PlotCentroid(data, keypoints):
    np_vals = np.array(data)
    selected_vals = []
    
    int_arg = stoi_map['spine_top'] - 1
    mid_x = np_vals[:,0,int_arg]
    mid_y = np_vals[:,1,int_arg]

    for point in keypoints:
        start_iter = (point - 1)
        selected = np_vals[:,:,start_iter]
        selected_vals.append(selected)
    
    selected_vals = np.stack(selected_vals, axis=2)
    np_means_avg = np.mean(selected_vals, axis=2)
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        plt.xlabel("x Position (m)")
        plt.ylabel("y Position (m)")
    else:
        plt.xlabel("x Pixel Position")
        plt.ylabel("y Pixel Position")
    shifted_x = [(x - mid_x) / scale for x in np_means_avg[:,0]]
    shifted_y = [(y - mid_y) / scale for y in np_means_avg[:,1]]
    
    plt.scatter(shifted_x, shifted_y, alpha=np_means_avg[:,2], s=1)

    plt.title("Centroid of movement")
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()

def PlotLocSpectrum(data, keypoints):
    x_dict = {}
    y_dict = {}
    fig, ax = plt.subplots(len(keypoints))
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
        name = itos_map[point]

        x_dict[name] = [(1280 - x) / scale for x in np_vals[:,0, start_iter]]
        y_dict[name] = np_vals[:,1, start_iter] / scale
    
    int_arg = stoi_map['spine_top'] - 1
    mid_x = [(1280 - x) / scale for x in np_vals[:,0, int_arg]]
    mid_y = np_vals[:,1,int_arg] / scale
    x_dict["center"] = mid_x
    y_dict["center"] = mid_y

    ax[0].boxplot(x_dict.values(), vert=False)
    ax[1].boxplot(y_dict.values())
    ax[0].set_yticklabels(x_dict.keys())
    ax[1].set_xticklabels(y_dict.keys())
    plt.show()

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

    for point in keypoints:
        start_iter = (point - 1)
        selected = np_vals[:,:,start_iter]
        if normalized:
            selected[:,0] = abs(mid_x - selected[:,0]) / scale
            selected[:,1] = abs(mid_y - selected[:,1]) / scale
        else:    
            selected[:,0] = (selected[:,0] - mid_x) / scale
            selected[:,1] = (selected[:,1] - mid_y) / scale

        plt.scatter(selected[:,0], selected[:,1], alpha=np_vals[:,2, start_iter], s=1)

    labels = [itos_map[x] for x in keypoints]
    plt.legend(labels)
    if normalized:
        plt.title("Normalized Distance from Center")
    else:
        plt.title("Distance from Center")
    
    if not normalized:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    plt.show()

vel_blocks = 15
def PlotVelocityHeatMap(data, keypoints):
    np_vals = np.array(data)
    do_iter = False
    if len(keypoints) > 1:    
        do_iter = True
    fig, ax = plt.subplots(len(keypoints))
    heat_map = np.zeros((720, 1280))
    scale = 1.0
    if video_pix_per_m > 0:
        scale = video_pix_per_m
        
    for i, point in enumerate(keypoints):
        start_iter = (point - 1)
        xs = np_vals[:,0, start_iter]
        xs /= vel_blocks
        xs = np.where(xs>(1279), int(1279), xs)
        ys = np_vals[:,1, start_iter]
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

        heat_map[y_ints.astype(int), x_ints.astype(int)] = total_vels

        if do_iter:
            im = ax[i].imshow(heat_map, cmap='cool', vmin = 0.0, vmax = 250.0)

            ax[i].set_title("Velocity of " + str(itos_map[point]))
            ax[i].set_xlabel("x Pixel Position")
            ax[i].set_ylabel("y Pixel Position")
            ax[i].set_xlim([0, 1280 / vel_blocks])
            ax[i].set_ylim([0, 720 / vel_blocks])
            fig.colorbar(im)
        else:
            im = ax.imshow(heat_map, cmap='cool', vmin = 0.0, vmax = 250.0)

            ax.set_title("Velocity of " + str(itos_map[point]))
            ax.set_xlabel("x Pixel Position")
            ax.set_ylabel("y Pixel Position")
            ax.set_xlim([0, 1280/ vel_blocks])
            ax.set_ylim([0, 720 / vel_blocks])
            fig.colorbar(im)
    plt.show()

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

        xs = np_vals[:,0, start_iter]
        xs = np.where(xs>1279, 1279, xs)
        ys = np_vals[:,1, start_iter]
        ys = np.where(ys>719, 719, ys)

        confs = np_vals[:,2, start_iter]
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
    plt.legend(labels)
    plt.title("Speed over time")
    plt.xlabel("seconds")
    
    plt.show()

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

        ax1.plot(np.arange(0, len(list_vels_x)/float(video_fps), 1.0/float(video_fps))[:len(avged_vels_x)], avged_vels_x)
        ax2.plot(np.arange(0, len(list_vels_y)/float(video_fps), 1.0/float(video_fps))[:len(avged_vels_y)], avged_vels_y)

    ax1.set_ylim([my_min, my_max])
    ax2.set_ylim([my_min, my_max])
    labels = [itos_map[x] for x in keypoints]
    labels.append("center")
    plt.legend(labels)
    ax1.set_title("X velocity over time")
    ax2.set_title("Y velocity over time")
    plt.xlabel("seconds")
    plt.show()


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
    print("showing aperture")
    plt.show()


def Plot(data, keypoints, type):
    match type:
        case PlotType.POINT_CLOUD:
            PlotPointCloud(data, keypoints)
            return True
        case PlotType.CENTROID:
            PlotCentroid(data, keypoints)
            return True
        case PlotType.POS_SPEC:
            PlotLocSpectrum(data, keypoints)
            return True
        case PlotType.CENT_DIST:
            PlotDistFromCenter(False, data, keypoints)
            return True
        case PlotType.CENT_DIST_NORM:
            PlotDistFromCenter(True, data, keypoints)
            return True
        case PlotType.VEL_HEAT_MAP:
            PlotVelocityHeatMap(data, keypoints)
            return True
        case PlotType.TOTAL_VEL_OVER_TIME:
            PlotSpeedOverTime(data, keypoints)
            return True
        case PlotType.VEL_OVER_TIME:
            PlotVelocitiesOverTime(data, keypoints)
            return True
        case PlotType.APERATURE:
            PlotAperatureOverTime(data, keypoints)
        case _:
            return False


if __name__ == '__main__':
    keypoint_names = ['head','spine_top','shoulder_right','elbow_right','wrist_right','shoulder_left',
    'elbow_left','wrist_left','spine_base','hip_right','knee_right','ankle_right','hip_left',
    'knee_left','ankle_left','eye_right','eye_left','ear_right','ear_left','big_toe_left',
    'little_toe_left','heel_left','big_toe_right','little_toe_right','heel_right','palm_left',
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

        if arg == '--aperture_over_time':
            gather_keypoints = True
            plot_type = PlotType.APERATURE

        if arg == '--fps':
            gather_keypoints = False
            video_fps = sys.argv[i+1]
        if arg == '--pix_to_m':
            gather_keypoints = False
            pix = float(sys.argv[i+1])
            m = float(sys.argv[i+2])
            video_pix_per_m = pix / m

    data = ReadData(file_path)
    flag = Plot(data, keypoints, plot_type)
    print("plotted")

    
    
    
    