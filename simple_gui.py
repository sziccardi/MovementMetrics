import PySimpleGUI as sg
import os.path
import json

import movement_metrics as mm
import scipy.signal as signal
import numpy as np
import scipy.signal as signal
import math

from PIL import Image, ImageTk
import io

from enum import Enum

graph_size = (520, 325)
long_graph_size = (1020, 200)
image_size = (475, 350)
global frame_size
frame_size = (1280, 720)
font = 'Ariel'
font_size_heading = 16
font_size_subheading = 12
font_size_regular = 10
colors = ["blue", "orange", "olivedrab", "slategrey", "purple", "red", "salmon", "light blue"]
window = None

class GraphType(Enum):
    LINE_GRAPH = 0
    POINT_GRAPH = 1


def get_main_layout():
    left_column = [
        [sg.Text("Plot Settings", font=(font, font_size_heading, 'bold'))],
        [sg.HSep()],
        [sg.HSep()],
        [sg.Text('Video to analyze:', font=(font, font_size_subheading), key="-SELECTED FILE-")],
        [sg.Button(button_text='Browse', size=(25,1),enable_events=True,key="-EXISTING VIDEO BUTTON-")],
        [sg.HSep()],
        [sg.Text('Track Points', font=(font, font_size_subheading))],
        [sg.Listbox(values=['head','shoulder_right','elbow_right','wrist_right','shoulder_left',
        'elbow_left','wrist_left'], font=(font, font_size_regular), enable_events=True, 
            size=(26,7), key="-TRACK POINT LIST-", select_mode='multiple'
        )], 
        [sg.HSep()],
        [sg.Text('Script settings', font=(font, font_size_heading, 'bold'))],
        [sg.HSep()],
        [sg.HSep()],
        [sg.Text("Camera frame rate", font=(font, font_size_subheading))],
        [sg.InputText("30", font=(font, font_size_regular), size=(5,1), key="-FPS-"), sg.Text("FPS", font=(font, font_size_regular))],
        [sg.HSep()],
        [sg.Text("Pixels to Meters", font=(font, font_size_subheading))],
        [sg.Text("*If blank, plot units are pixels*", font=(font, font_size_subheading))],
        [sg.InputText(size=(8,1), key="-PIX SCALE-", font=(font, font_size_regular)), sg.Text("Pixels", font=(font, font_size_regular))],
        [sg.HSep()],
        [sg.Text("Smoothing amount", font=(font, font_size_subheading))],
        [sg.InputText("1", font=(font, font_size_regular), size=(5,1), key="-CONV WIDTH-"), sg.Text("Frames", font=(font, font_size_regular))],
        [sg.HSep()],
        [sg.Button(button_text='PLOT', font=(font, font_size_subheading), size=(15,1),enable_events=True,key="-RUN SCRIPT-", visible=False)],
        ]

    image_column = [
        [sg.Text("Processed video will show up here", font=(font, font_size_subheading), key="-IMAGE TITLE-")],
        [sg.Image(filename="placeholder.png", size=image_size, key='-FRAME IMAGE-')],
        [sg.Graph(canvas_size=(image_size[0],7), graph_bottom_left=(0, 0), graph_top_right=(image_size[0], 5), change_submits=True, drag_submits=True, background_color="white", float_values = True, key="-FRAME HIGHLIGHT BAR-")],
        [sg.Slider(range=(0, 100), default_value=0, disable_number_display=True, orientation='horizontal', size=(53,7), key="-SCRUB BAR-", visible=False, enable_events=True)],
        [sg.Button(button_text='Prev Key Frame', font=(font, font_size_regular), enable_events=True, key="-LEFT FRAME-"),sg.Text("", key="-SELECTED FRAMES-", font=(font, font_size_regular)), sg.Button(button_text='Next Key Frame', font=(font, font_size_regular), enable_events=True, key="-RIGHT FRAME-")],
        [sg.Text("And here", font=(font, font_size_subheading), justification = 'left', key="-OVER TIME PLOT TITLE-")],
    ]

    plot_column = [
        [sg.Text("Plotted data will show up here", font=(font, font_size_subheading), key="-GENERAL PLOT TITLE-")],
        [sg.Graph(canvas_size=graph_size, graph_bottom_left=(0, 0), graph_top_right=graph_size, background_color="white", float_values = True, key="-PLOT CANVAS-", change_submits=True, drag_submits=True)],
        [sg.Graph(canvas_size=(graph_size[0], 20), graph_bottom_left=(0, 0), graph_top_right=(graph_size[0]-95, 25), background_color="white", float_values = True, key="-PLOT LEGEND-")]
    ]

    main_column = [[sg.Text('Plots', font=(font, font_size_heading, 'bold'))],
    [sg.HSep()],
    [sg.HSep()],
    [sg.Column(image_column, key="-FRAME COLUMN-", element_justification='c'), sg.Column(plot_column, key="-PLOT COLUMN-", element_justification='c')],
    [sg.Graph(canvas_size=(long_graph_size[0], long_graph_size[1]), graph_bottom_left=(0, 0), graph_top_right=(long_graph_size[0], long_graph_size[1]), change_submits=True, drag_submits=True, background_color="white", float_values = True, key="-OVER TIME PLOT 1-")],
    [sg.Graph(canvas_size=(long_graph_size[0], long_graph_size[1]), graph_bottom_left=(0, 0), graph_top_right=(long_graph_size[0], long_graph_size[1]), change_submits=True, drag_submits=True, background_color="white", float_values = True, key="-OVER TIME PLOT 2-", visible=False)]   
    ]
    right_column = [[sg.Text('Metrics:', font=(font, font_size_heading, 'bold'))],
    [sg.HSep()],
    [sg.HSep()],
    [sg.Multiline("", font=(font, font_size_regular), expand_x=True, expand_y=True, disabled=True, size=(20,38), key="-COMPUTED METRICS-")]
    
    ]

    layout = [
        [
            sg.Column(left_column, key="-MAIN OPTIONS COL-", size=(210, 650)),
            sg.VSeperator(),
            sg.Column(main_column, key="-MAIN PLOT DISPLAY-", size=(1040, 650)),
            sg.VSeperator(),
            sg.Column(right_column, key="-MAIN SCRIPT SETTINGS-", size=(150,650))
        ]
    ]
    return layout

def get_frame_select_layout():
    file_list_column = [
        [sg.Text("Which folder contains the pose and frame files?")],
        [ 
            sg.In(size=(55, 1), enable_events=True, key="-FOLDER-"), 
            sg.FolderBrowse()
        ],
        [sg.Button(button_text='Next',size=(5,1),enable_events=True,key="-NEXT BUTTON-")]
    ]

    layout = [
        [
            sg.Column(file_list_column)
        ]
    ]

    return layout

def display_file_select(chosen_file):
    my_layout = get_frame_select_layout() 
    window_name = "Processed Folder Selection"
    sub_window = sg.Window(window_name, my_layout, modal=True)
    file_loc = ""
    while True:
        event, values = sub_window.read()
        if event == "-NEXT BUTTON-":
            break
        elif event == "-BACK BUTTON-":
            chosen_file = ''
            break
        elif event == "-FOLDER-":
            file_loc = values["-FOLDER-"]        
        elif event == "-FILE LIST OPTIONS-":  # A file was chosen from the listbox
            try:
                chosen_file = values["-FILE LIST OPTIONS-"][0]
            except:
                pass
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    sub_window.close()
    return file_loc, chosen_file


def draw_axes(graph, ax_lims, axes_labels, scale_axes, custom_tick_count_x = None, custom_tick_count_y = None):
    #                    0        1        2        3        4        5        6        7
    #ax_lims order = [x_max_0, x_max_1, y_max_0, y_max_1, x_min_0, x_min_1, y_min_0, y_min_1]
    x_min = min(ax_lims[4], ax_lims[5])
    x_max = max(ax_lims[0], ax_lims[1])
    y_min = min(ax_lims[6], ax_lims[7])
    y_max = max(ax_lims[2], ax_lims[3])

    x_range = x_max - x_min
    y_range = y_max - y_min

    if custom_tick_count_x:
        x_tick_count = custom_tick_count_x
    else:
        x_tick_count = x_range/7.0
        if x_range < 7:
            x_tick_count = 1

    if custom_tick_count_y:
        y_tick_count = custom_tick_count_y
    else:
        y_tick_count = y_range/7.0
        if y_range < 7:
            y_tick_count = 1
    
   

    scaled_y_min = y_min 
    scaled_x_min = x_min
    scaled_y_max = y_max
    scaled_x_max = x_max
    if scale_axes:
        scaled_y_min = y_min-y_tick_count
        scaled_x_min = x_min-x_tick_count
        scaled_y_max = y_max+y_tick_count
        scaled_x_max = x_max+x_tick_count
    
    dot_size=1 #x_range/150

    graph.change_coordinates((scaled_x_min, scaled_y_min),(scaled_x_max, scaled_y_max))
    
    #draw axes
    ax_x_min = x_min if scaled_x_min > 0 else scaled_x_min
    ax_x_max = x_max if scaled_x_max < 0 else scaled_x_max
    ax_y_min = y_min if scaled_y_min > 0 else scaled_y_min
    ax_y_max = y_max if scaled_y_max < 0 else scaled_y_max

    x_ax_label_pos = y_ax_label_pos = 0
    x_ax_label_anch = y_ax_label_anch = "center"

    label_angle = 0
    if scaled_x_min > 0:
        x_ax_label_pos = x_range/2 + ax_x_min
    else:
        if abs(scaled_x_min) > abs(scaled_x_max):
            x_ax_label_pos = ax_x_min + x_range/75
            x_ax_label_anch = sg.TEXT_LOCATION_TOP_LEFT
        else:
            x_ax_label_pos = ax_x_max - x_range/75
            x_ax_label_anch = sg.TEXT_LOCATION_TOP_RIGHT

    if scaled_y_min > 0:
        y_ax_label_pos = y_range/2 + ax_y_min
        label_angle = 90
    else:
        if abs(scaled_y_min) > abs(scaled_y_max):
            y_ax_label_pos = ax_y_min + y_range/75
            y_ax_label_anch = sg.TEXT_LOCATION_BOTTOM_LEFT
        else:
            y_ax_label_pos = ax_y_max - y_range/75
            y_ax_label_anch = sg.TEXT_LOCATION_TOP_LEFT

    h_tick_len = x_range/60.0
    v_tick_len = y_range/60.0
    x_shift = 3.5*h_tick_len
    if label_angle != 0:
        x_shift = -4.5*h_tick_len

    graph.draw_line(
        (ax_x_min, max(0, scaled_y_min + y_tick_count)), 
        (ax_x_max, max(0, scaled_y_min + y_tick_count)), color="black", width=dot_size*0.75) #x axis
        
    for x in range(int(ax_x_min), int(ax_x_max), int(x_tick_count)):
        graph.draw_line((x, max(0, scaled_y_min + y_tick_count)-v_tick_len), (x, max(0, scaled_y_min + y_tick_count)+v_tick_len))  #Draw a scale
        if x != 0:
            graph.draw_text(str(x), (x, max(0, scaled_y_min + y_tick_count)-2.5*v_tick_len), color='black')  #Draw the value of the scale
    
    graph.draw_text(axes_labels[0], (x_ax_label_pos, max(0, scaled_y_min + y_tick_count)-4.75*v_tick_len), text_location = x_ax_label_anch, color="black")

    graph.draw_line(
        (max(0, scaled_x_min + x_tick_count), ax_y_min), 
        (max(0, scaled_x_min + x_tick_count), ax_y_max), color="black", width=dot_size*0.75) #y axis

    
    for y in range(int(ax_y_min), int(ax_y_max), int(y_tick_count)):
        graph.draw_line((max(0, scaled_x_min + x_tick_count)-h_tick_len, y), (max(0, scaled_x_min + x_tick_count)+h_tick_len, y))
        if y != 0:
            graph.draw_text(str(y), (max(0, scaled_x_min + x_tick_count)-2.25*h_tick_len, y), color='black')
    
    graph.draw_text(axes_labels[1], (max(0, scaled_x_min + x_tick_count)+x_shift, y_ax_label_pos), angle=label_angle, text_location = y_ax_label_anch, color="black")
    return dot_size

def draw_legend(graph, labels, colors):
    for i, dot_color in enumerate(colors):
        graph.draw_point((15 + i*55, 12.5), 10, color=dot_color)
        name = labels[i].replace("_", "\n")
        graph.draw_text(name, (25 + i*55, 0.5), font=("Arial", 6), text_location = sg.TEXT_LOCATION_BOTTOM_LEFT, color="black")

def create_basic_plot(graph, data, data_labels, legend, axes_labels, graph_type):
    #ax_lims order = [x_max_0, x_max_1, y_max_0, y_max_1, x_min_0, x_min_1, y_min_0, y_min_1]
    ax_lims = [-10000, -10000, -10000, -10000, 10000, 10000, 10000, 10000]
    for key in range(len(data)):
        vals_plot_0 = list(data[key].values())

    max_x = max([point[0] for point in vals_plot_0])
    max_y = max([point[1] for point in vals_plot_0])
    min_x = min([point[0] for point in vals_plot_0])
    min_y = min([point[1] for point in vals_plot_0])
    
    if ax_lims[0] < max_x:
        ax_lims[0] = max_x
    if ax_lims[2] < max_y:
        ax_lims[2] = max_y
    if ax_lims[4] > min_x:
        ax_lims[4] = min_x
    if ax_lims[6] > min_y:
        ax_lims[6] = min_y

    conf_thresh = mm.GetPlotSpecificInfo("relative position")[0]
    
    dot_size = draw_axes(graph, ax_lims, axes_labels, True)
    #color_select = random.sample(colors, len(data))
    if graph_type == GraphType.LINE_GRAPH: #must be angles over time
        for key in range(len(data)):
            np_data = np.array(list(data[key].values()))
            y_peaks = signal.find_peaks(np_data[:,1], threshold=30.0)
            if len(y_peaks[0]) > 0:
                #print("I found ", len(y_peaks[0]), " weird peaks in the y values of point cloud at ", y_peaks)
                for peak in y_peaks[0]:
                    if peak > 0 and peak < np_data.shape[0]-1:
                        #print("peak is ", np_data[peak,1])
                        #print("neighbors are ", np_data[peak-1,1], " and ", np_data[peak+1,1])
                        val = (np_data[peak-1,1] + np_data[peak+1,1])/2.0
                        #print("so I'm replacing ", np_data[peak,1], " with ", val)
                        data[key][peak][1] = val

        line_color = colors[key]
        for point in range(len(data[key])-1):
            if data[key][point][2] > mm.GetPlotSpecificInfo("angles over time")[0]:
                graph.draw_line((data[key][point][0], data[key][point][1]), (data[key][point+1][0], data[key][point+1][1]), width=dot_size, color=line_color)
    elif graph_type == GraphType.POINT_GRAPH: #must be point cloud
        for key in range(len(data)):
            np_data = np.array(list(data[key].values()))
            x_peaks = signal.find_peaks(np_data[:,0], threshold=100.0)
            if len(x_peaks[0]) > 0:
                for peak in x_peaks[0]:
                    if np_data[peak,2] > conf_thresh and np_data[peak-1,2] > conf_thresh and np_data[peak+1,2] > conf_thresh and peak > 0 and peak < np_data.shape[0]-1:
                        val = (np_data[peak-1][0] + np_data[peak+1,0])/2.0
                        data[key][peak][0] = val
                
            y_peaks = signal.find_peaks(np_data[:,1], threshold=100.0)
            if len(y_peaks[0]) > 0:
                for peak in y_peaks[0]:
                    if np_data[peak,2] > conf_thresh and np_data[peak-1,2] > conf_thresh and np_data[peak+1,2] > conf_thresh and peak > 0 and peak < np_data.shape[0]-1:
                        val = (np_data[peak-1,1] + np_data[peak+1,1])/2.0
                        data[key][peak][1] = val


            for point in range(len(data[key])):
                if data[key][point][2] > mm.GetPlotSpecificInfo("relative position")[0]:
                    graph.draw_point((data[key][point][0], data[key][point][1]), dot_size, color=colors[key])
    
    draw_legend(legend, data_labels, colors[:len(data)])

def create_two_plots(graphs, data, data_labels, legend, axes_labels, graph_type):
    ax_lims = [-10000, -10000, -10000, -10000, 10000, 10000, 10000, 10000]
    
    for key in range(len(data)):
        vals_plot_0 = data[key][0]
        vals_plot_1 = data[key][1]
        max_x_0 = max([vals_plot_0[key_point][0] for key_point in vals_plot_0])
        max_y_0 = max([vals_plot_0[key_point][1] for key_point in vals_plot_0])
        min_x_0 = min([vals_plot_0[key_point][0] for key_point in vals_plot_0])
        min_y_0 = min([vals_plot_0[key_point][1] for key_point in vals_plot_0])
        max_x_1 = max([vals_plot_1[key_point][0] for key_point in vals_plot_1])
        max_y_1 = max([vals_plot_1[key_point][1] for key_point in vals_plot_1])
        min_x_1 = min([vals_plot_1[key_point][0] for key_point in vals_plot_1])
        min_y_1 = min([vals_plot_1[key_point][1] for key_point in vals_plot_1])
        
        if ax_lims[0] < max_x_0:
            ax_lims[0] = max_x_0
        if ax_lims[2] < max_y_0:
            ax_lims[2] = max_y_0
        if ax_lims[4] > min_x_0:
            ax_lims[4] = min_x_0
        if ax_lims[6] > min_y_0:
            ax_lims[6] = min_y_0
        if ax_lims[1] < max_x_1:
            ax_lims[1] = max_x_1
        if ax_lims[3] < max_y_1:
            ax_lims[3] = max_y_1
        if ax_lims[5] > min_x_1:
            ax_lims[5] = min_x_1
        if ax_lims[7] > min_y_1:
            ax_lims[7] = min_y_1
    
    dot_size = draw_axes(graphs[0], ax_lims, axes_labels[0], True)
    dot_size = draw_axes(graphs[1], ax_lims, axes_labels[1], True)
    conf_thresh = mm.GetPlotSpecificInfo("relative position")[0]
    if graph_type == GraphType.LINE_GRAPH: # must be relative pos over time
        
        for key in range(len(data)):
            plot_data = data[key]
            line_color = colors[key]
            for plot in range(len(plot_data)):
                np_data = np.array(list(plot_data[plot].values()))
                y_peaks = signal.find_peaks(np_data[:,1], threshold=100.0)
                if len(y_peaks[0]) > 0:
                    #print("I found ", len(y_peaks[0]), " weird peaks in plot #", plot, " of relative pos over time at ", y_peaks)
                    for peak in y_peaks[0]:
                        if np_data[peak,2] > conf_thresh and np_data[peak-1,2] > conf_thresh and np_data[peak+1,2] > conf_thresh and peak > 0 and peak < len(np_data[:,1])-1:
                            #print("peak is ", np_data[peak,1])
                            #print("neighbors are ", np_data[peak-1,1], " and ", np_data[peak+1,1])
                            val = (np_data[peak-1,1] + np_data[peak+1,1])/2.0
                            #print("replacing ", np_data[peak,1], " with ", val)
                            plot_data[key][peak][1] = val
                    
                for point in range(len(plot_data[plot])-1):
                    if np_data[point,2] > conf_thresh and np_data[point+1,2] > conf_thresh:
                        graphs[plot].draw_line((plot_data[plot][point][0], plot_data[plot][point][1]), (plot_data[plot][point+1][0], plot_data[plot][point+1][1]), color=line_color, width=dot_size)
    elif graph_type == GraphType.POINT_GRAPH: #not used anumore
        for key in range(len(data)):
            plot_data = data[key]
            line_color = colors[key]
            for plot in range(len(plot_data)):
                for point in range(len(plot_data[plot])):
                    if plot_data[plot][point][2] > mm.GetPlotSpecificInfo("relative position")[0]:
                        graphs[plot].draw_point((plot_data[plot][point][0], plot_data[plot][point][1]), dot_size, color=line_color)

    draw_legend(legend, data_labels, colors[:len(data)])


def read_frame_files(file_loc):
    try:
        file_list = os.listdir(file_loc+'/video_frames')
    except:
        file_list = []
    file_list = [val for val in file_list if val.lower().endswith((".jpg")) or val.lower().endswith((".png"))]
    frames = [(file_loc+'/video_frames/'+val) for val in file_list]
    frames.sort()
    file_list.sort()
    
    keys= [ int(filename[filename.find('_')+1:filename.find('_')+1+filename[filename.find('_')+1:].find('.')]) -1 for filename in file_list]
    
    frames_dict = {keys[i]: frames[i] for i in range(len(keys))}
    
    return frames_dict

def read_pose_files(file_loc):
    try:
        file_list = os.listdir(file_loc+'/pose_info')
    except:
        file_list = []
    
    chosen_files = [val for val in file_list if val.lower().endswith((".json"))]
    chosen_files = sorted(chosen_files)
    return chosen_files

def read_pose_data(files):
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
                        x_dif = frame_size[0]/2.0 - json_file['people'][person]['pose_keypoints_2d'][0]
                        y_dif = frame_size[1]/2.0 - json_file['people'][person]['pose_keypoints_2d'][1]
                        if len(json_file['people'][person]['pose_keypoints_2d']) > 1:
                            x_dif = frame_size[0]/2.0 - json_file['people'][person]['pose_keypoints_2d'][0]
                            y_dif = frame_size[1]/2.0 - json_file['people'][person]['pose_keypoints_2d'][1]
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


def get_img_data(f, maxsize=image_size, first=False):
    img = Image.open(f)
    global frame_size
    frame_size = img.size
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def display_frame(i):
    if i is not None:
        window['-FRAME IMAGE-'].update(data=get_img_data(frames[i], first=False))
        img_name = frames[i][file_loc.rfind('/')+1:]
        window['-IMAGE TITLE-'].update(value=img_name)

        frame_loc = i / len(frames)
        frame_loc = frame_loc * image_size[0]
        id = window["-FRAME HIGHLIGHT BAR-"].draw_rectangle((frame_loc, 7), (frame_loc+1, 0), fill_color='black', line_color="black")
        return id
    


def display_metrics(data, labels):
    total_track_point_text = ""
    fps = (float)(values["-FPS-"])

    for i, name in enumerate(labels):
        data_list = list(data[i].values())
        np_data = np.array(data_list)
        temp_total_count, temp_num_count = mm.getAxesCrossedCounts(np_data[:,0], ("right" in name))
        total_track_point_text = total_track_point_text + "\n" + name + " : \n - crossed body midline " + str(temp_num_count) + " times\n - " + str(round(temp_total_count / fps,2)) + " sec spent crossed\n"
        
        temp_total_count, temp_num_count = mm.getAxesCrossedCounts(np_data[:,1], True)
        total_track_point_text = total_track_point_text + " - raised above shoulders " + str(temp_num_count) + " times\n - " + str(round(temp_total_count / fps,2)) + " sec spent raised\n"
        
        conf = mm.GetPlotSpecificInfo("relative position")[0]
        data_filter = np_data[:,2] > conf

        data_x_mean = mm.getMean(np_data[data_filter,0])
        data_y_mean = mm.getMean(np_data[data_filter,1])
        data_x_var = mm.getSTD(np_data[data_filter,0])
        data_y_var = mm.getSTD(np_data[data_filter,1])
        total_track_point_text = total_track_point_text + " - average position ( " + str(round(data_x_mean,2)) + ", " + str(round(data_y_mean,2)) + " )\n"
        total_track_point_text = total_track_point_text + " - with std of ( "+ str(round(data_x_var,2)) + ", " + str(round(data_y_var,2)) + " )\n"
        
        filter = np_data[:,2] > conf
        included = sum(filter)
        num_skipped = np_data[:,2].shape[0] - included
        selected = np_data[filter]
        avg_conf = np.mean(selected[:,2])
        total_track_point_text = total_track_point_text + "# frames skipped: "+str(num_skipped) + "\n"
        total_track_point_text = total_track_point_text + "Average confidence: "+str(round(avg_conf,2)) + "\n"

        window['-COMPUTED METRICS-'].update(value=total_track_point_text)
        print(total_track_point_text)

if __name__ == '__main__':
    
    current_layout = get_main_layout()
    window = sg.Window(title="Bilateral Motion Analysis Toolkit (BMAT)", layout=current_layout)
    
    #mocap file select variables
    chosen_file = ''
    chosen_files = []

    #mocap video frames
    current_frame = 0
    frames = []
    highlight = []
    highlight_marks = []
    current_frame_mark = None
    
    #plotting variables
    graph = window.Element("-PLOT CANVAS-")
    long_graph = window.Element("-OVER TIME PLOT 1-")
    long_graph2 = window.Element("-OVER TIME PLOT 2-")
    dragging = False
    start_point = end_point = prior_plot = None
    prior_rect = (None, None)

    #event loop
    file_loc = ""
    while current_layout:
        event, values = window.read()
        current_process = None
        #overall events
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == "-EXISTING VIDEO BUTTON-":
            file_loc, chosen_file = display_file_select(chosen_file)
            chosen_files.clear()
            frames.clear()
            chosen_files = read_pose_files(file_loc)
            loc_name = file_loc[file_loc.rfind('/')+1:]
            frames = read_frame_files(file_loc)
            if len(frames) == 0:
                print("WARNING: No video frames found for the selected folder.")
            current_frame = list(frames.keys())[0]

            loc_name = file_loc[file_loc.rfind('/')+1:]
            window["-SELECTED FILE-"].update(value="Currently selected: " + loc_name, visible=True)
            window.TKroot.attributes('-topmost', 1)
            window.TKroot.attributes('-topmost', 0)

        #lets run plotting!
        if event == "-RUN SCRIPT-":
            window['-COMPUTED METRICS-'].Widget.config(wrap='word')
            
            real_files = [os.path.join(file_loc+'/pose_info', f) for f in chosen_files]
            real_data = read_pose_data(real_files)
            
            track_points = values["-TRACK POINT LIST-"]

            if len(frames) > 0:
                window["-SCRUB BAR-"].update(visible=True, range=(0, len(frames)-1))

                window['-FRAME IMAGE-'].update(data=get_img_data(frames[current_frame], first=True))
                img_name = frames[current_frame][file_loc.rfind('/')+1:]
                window['-IMAGE TITLE-'].update(value=img_name)
            else:
                window['-FRAME IMAGE-'].update(filename="placeholder.png", size=image_size)
                window['-IMAGE TITLE-'].update(value="No video frames found")
            
            fps = (int)(values["-FPS-"])
            pix_scale = ''
            if values["-PIX SCALE-"] != '':
                pix_scale = (float)(values["-PIX SCALE-"])
            else:
                pix_scale = -1
            cov_w = (int)(values["-CONV WIDTH-"])
            
            data1, labels1, ax_labels1 = mm.run_script_get_data(real_files, frame_size, "relative position", track_points, fps, pix_scale, cov_w)
            data2, labels2, ax_labels2 = mm.run_script_get_data(real_files, frame_size, "relative position over time", track_points, fps, pix_scale, cov_w)
            display_metrics(data1, labels1)

            graph.Erase()
            
            long_graph.Erase()
            long_graph2.Erase()
            window["-PLOT LEGEND-"].Erase()
            
            title1 = "RELATIVE POSITION POINT CLOUD"
            title2 = "RELATIVE POSITIONS OVER TIME"
            
            window["-GENERAL PLOT TITLE-"].update(value=title1)
            window["-OVER TIME PLOT TITLE-"].update(value=title2)
            
            window["-PLOT CANVAS-"].set_size(graph_size)
            create_basic_plot(graph, data1, labels1, window["-PLOT LEGEND-"], ax_labels1, GraphType.POINT_GRAPH)

            window["-OVER TIME PLOT 2-"].update(visible=True)
            window["-OVER TIME PLOT 1-"].set_size((long_graph_size[0], long_graph_size[1]/2-3))
            window["-OVER TIME PLOT 2-"].set_size((long_graph_size[0], long_graph_size[1]/2-3))
            create_two_plots([long_graph, long_graph2], data2, labels2, window["-PLOT LEGEND-"], ax_labels2, GraphType.LINE_GRAPH)

    
        if len(chosen_files) > 0 and len(values["-TRACK POINT LIST-"]) > 0:
            window["-RUN SCRIPT-"].update(visible=True)
        else:
            window["-RUN SCRIPT-"].update(visible=False)

        if event == "-PLOT CANVAS-" or event == "-OVER TIME PLOT 1-" or event == "-OVER TIME PLOT 2-" or event == "-FRAME HIGHLIGHT BAR-":
            x, y = values[event]
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect[1]:
                window[prior_rect[0]].delete_figure(prior_rect[1])
                prior_rect = (None, None)
            if None not in (start_point, end_point):
                prior_rect = (event, window[event].draw_rectangle(start_point, end_point, line_color='red'))
                

        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            highlight.clear()
            highlight = []

            if start_point is not None and end_point is not None:

                min_x = min(start_point[0], end_point[0])
                max_x = max(start_point[0], end_point[0])
                min_y = min(start_point[1], end_point[1])
                max_y = max(start_point[1], end_point[1])


            plot_specfic_data1 = data1
            plot_specfic_data2 = data2
            
            if prior_rect[0] == "-OVER TIME PLOT 1-":
                plot_specfic_data2 = data2[0]
            elif prior_rect[0] == "-OVER TIME PLOT 2-":
                plot_specfic_data2 = data2[1]
            
            max_points = 0
            
            if plot_specfic_data1 is not None:
                for key in range(len(plot_specfic_data1)):
                    if len(plot_specfic_data1[key]) > max_points:
                        max_points = len(plot_specfic_data1[key])

                if plot_specfic_data2 is not None:
                    for key in range(len(plot_specfic_data2)):
                        if len(plot_specfic_data2[key]) > max_points:
                            max_points = len(plot_specfic_data2[key])

                
                #highlight = [0 for i in range(max_points)]
                conf_thresh = mm.GetPlotSpecificInfo("relative position")[0]
                if prior_rect[0] == "-FRAME HIGHLIGHT BAR-":
                    min_frame = min_x / image_size[0]
                    min_frame = int(min_frame*len(frames))
                    print("min frame ", min_frame)

                    max_frame = max_x / image_size[0]
                    max_frame = int(max_frame*len(frames))
                    print("max frame ", max_frame)

                    highlight = list(frames.keys())[min_frame:max_frame]
                elif plot_specfic_data2 is not None and prior_rect[0] == "-OVER TIME PLOT 1-" or prior_rect[0] == "-OVER TIME PLOT 2-":
                    frame_list = list(frames.keys())
                    for key in range(len(plot_specfic_data2)):
                        for point in range(len(plot_specfic_data2[key])):
                            conf = plot_specfic_data2[key][point][2] > conf_thresh
                            within_x = plot_specfic_data2[key][point][0] > min_x and plot_specfic_data2[key][point][0] < max_x
                            within_y = plot_specfic_data2[key][point][1] > min_y and plot_specfic_data2[key][point][1] < max_y
                            if conf and within_x and within_y:
                                highlight.append(frame_list[point])
                elif plot_specfic_data1 is not None:
                    frame_list = list(frames.keys())
                    for key in range(len(plot_specfic_data1)):
                        for point in range(len(plot_specfic_data1[key])):
                            conf = plot_specfic_data1[key][point][2] > conf_thresh
                            within_x = plot_specfic_data1[key][point][0] > min_x and plot_specfic_data1[key][point][0] < max_x
                            within_y = plot_specfic_data1[key][point][1] > min_y and plot_specfic_data1[key][point][1] < max_y
                            if conf and within_x and within_y:
                               highlight.append(frame_list[point])
                
                
                #highlight_frames_iter = [i for i in range(max_points) if highlight[i]]

                print(f"grabbed rectangle from {start_point} to {end_point}")
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False
                
                if len(highlight) > 0:
                    current_frame = highlight[0]
                    print("CURRENT FRAME:", current_frame)
                    window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
                    current_frame_mark = display_frame(current_frame)
                    window["-SCRUB BAR-"].update(value=current_frame)
                    window["-SELECTED FRAMES-"].update(value="Selected keyframe 1 / "+str(len(highlight)))
                    
                    for mark_i in highlight_marks:
                        window["-FRAME HIGHLIGHT BAR-"].delete_figure(mark_i)
                    highlight_marks.clear()
                    window["-FRAME HIGHLIGHT BAR-"].Erase()
                    for frame_i in highlight:
                        frame_loc = frame_i / len(frames)
                        frame_loc = frame_loc * image_size[0]
                        id = window["-FRAME HIGHLIGHT BAR-"].draw_rectangle((frame_loc, 7), (frame_loc+1, 0), fill_color='red', line_color="red")
                        highlight_marks.append(id)
                else:
                    window["-SELECTED FRAMES-"].update(value="No keyframes selected")
            
            
        if event == "-LEFT FRAME-":
            try:
                index = highlight.index(current_frame)
                index = index - 1
            except:
                index = 0

            if index < 0:
                index = len(highlight) - 1
            
            if len(highlight) == 0:
                current_frame = None
            else:
                current_frame = highlight[index]
                window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
                current_frame_mark = display_frame(current_frame)
                window["-SCRUB BAR-"].update(value=current_frame)
                window["-SELECTED FRAMES-"].update(value="Selected keyframe " + str(index + 1)+" / "+str(len(highlight)))

        if event == "-RIGHT FRAME-":
            try:
                index = highlight.index(current_frame)
                index = index + 1
            except:
                index = 0

            if index > len(highlight) - 1:
                index = 0
            
            if len(highlight) == 0:
                current_frame = None
            else:
                current_frame = highlight[index]
                window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
                current_frame_mark = display_frame(current_frame)
                window["-SCRUB BAR-"].update(value=current_frame)
                window["-SELECTED FRAMES-"].update(value="Selected keyframe " + str(index + 1)+" / "+str(len(highlight)))


        #video events
        if event == "-SCRUB BAR-":
            
            current_frame = int(values['-SCRUB BAR-'])
            window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
            current_frame_mark = display_frame(current_frame)
            window["-SELECTED FRAMES-"].update(value="Not one of the "+str(len(highlight)) + " selected keyframes")

   
    window.close()