import PySimpleGUI as sg
import os.path

import movement_metrics as mm
import numpy as np
import scipy.signal as signal

from PIL import Image, ImageTk
import io

from enum import Enum

graph_size = (475, 325)
long_graph_size = (980, 200)
image_size = (475, 350)
global frame_size
frame_size = (1280, 720)
colors = ["blue", "orange", "olivedrab", "slategrey", "purple", "red", "salmon", "light blue"]
window = None

class GraphType(Enum):
    LINE_GRAPH = 0
    POINT_GRAPH = 1


def get_main_layout():
    left_column = [
        [sg.Text("Plot Settings")],
        [sg.HSep()],
        [sg.HSep()],
        [sg.Text('Currently selected:', key="-SELECTED FILE-")],
        #[sg.Text(key="-SELECTED FILE-", size=(30,1), enable_events=False, visible=False)], 
        #[sg.Button(button_text='Browse New', size=(23,1),enable_events=True,key="-NEW VIDEO BUTTON-"), sg.Image(data=processing_gif, size=(2,2), key="-PROCESSING GIF-", visible=False)], 
        [sg.Button(button_text='Browse Existing', size=(30,1),enable_events=True,key="-EXISTING VIDEO BUTTON-")],
        [sg.HSep()],
        [sg.Text('Track Points')],
        [sg.Listbox( #'spine_top','spine_base','hip_right','knee_right','ankle_right','hip_left',
        #'knee_left','ankle_left','eye_right','eye_left','ear_right','ear_left','big_toe_left',
        #'little_toe_left','heel_left','big_toe_right','little_toe_right','heel_right',,'palm_left',
        #'thumb_base_left','thumb_1_left','thumb_2_left','thumb_tip_left','pointer_base_left',
        #'pointer_1_left','pointer_2_left','pointer_tip_left','middle_base_left','middle_1_left',
        #'middle_2_left','middle_tip_left','ring_base_left','ring_1_left','ring_2_left','ring_tip_left',
        #'pinky_base_left','pinky_1_left','pinky_2_left','pinky_tip_left','palm_right','thumb_base_right',
        #'thumb_1_right','thumb_2_right','thumb_tip_right','pointer_base_right','pointer_1_right',
        #'pointer_2_right','pointer_tip_right','middle_base_right','middle_1_right','middle_2_right',
        #'middle_tip_right','ring_base_right','ring_1_right','ring_2_right','ring_tip_right',
        #'pinky_base_right','pinky_1_right','pinky_2_right','pinky_tip_right'
        values=['head','shoulder_right','elbow_right','wrist_right','shoulder_left',
        'elbow_left','wrist_left'], enable_events=True, 
            size=(30,5), key="-TRACK POINT LIST-", select_mode='multiple'
        )], 
        # [sg.HSep()],
        # [sg.Text('Available Plot Types')],
        # [ 
        #     sg.Listbox(
        #         #"point cloud", "centroid of motion", "position spectrum", "distance from center", 
        #         #"normalized distance from center", "speed over time", "velocity over time"
        #         #"relative position", "relative position over time", "movement heatmap", "angles over time", "angle histogram"
        #         values=["relative position", "relative angles"], 
        #         enable_events=True, size=(30,5), key="-PLOT LIST-"
        #     )
        # ],
        [sg.HSep()],
        [sg.Text('Script settings')],
        [sg.HSep()],
        [sg.HSep()],
        [sg.Text("Camera frame rate")],
        [sg.InputText("30", size=(5,1), key="-FPS-"), sg.Text("FPS")],
        [sg.HSep()],
        [sg.Text("Number of Pixels that make up 1 meter")],
        [sg.Text("*If blank, plot units are pixels*")],
        [sg.InputText(size=(8,1), key="-PIX SCALE-"), sg.Text("Pixels")],
        [sg.HSep()],
        [sg.Text("Smoothing amount or Convolution size")],
        [sg.InputText("1", size=(5,1), key="-CONV WIDTH-"), sg.Text("Frames")],
        [sg.HSep()],
        [sg.Button(button_text='PLOT', size=(30,1),enable_events=True,key="-RUN SCRIPT-", visible=False)],
        ]

    image_column = [
        [sg.Text("Processed video will show up here", key="-IMAGE TITLE-")],
        [sg.Image(filename="placeholder.png", size=image_size, key='-FRAME IMAGE-')],
        [sg.Graph(canvas_size=(image_size[0],7), graph_bottom_left=(0, 0), graph_top_right=(image_size[0], 5), change_submits=True, drag_submits=True, background_color="white", float_values = True, key="-FRAME HIGHLIGHT BAR-")],
        [sg.Slider(range=(0, 100), default_value=0, disable_number_display=True, orientation='horizontal', size=(53,7), key="-SCRUB BAR-", visible=False, enable_events=True)],
        [sg.Button(button_text='Prev Key Frame', enable_events=True, key="-LEFT FRAME-"), sg.Button(button_text='Next Key Frame', enable_events=True, key="-RIGHT FRAME-")],
        [sg.Text("", key="-SELECTED FRAMES-")],
    ]

    plot_column = [
        [sg.Text("Plotted data will show up here", key="-GENERAL PLOT TITLE-")],
        [sg.Graph(canvas_size=graph_size, graph_bottom_left=(0, 0), graph_top_right=graph_size, background_color="white", float_values = True, key="-PLOT CANVAS-", change_submits=True, drag_submits=True)],
        # [sg.Text("Export plots as:", key="-EXPORT TEXT 1-"), 
        # sg.InputText(size=(20,1), key="-PLOT NAME-"), 
        # sg.Text(".png", key="-EXPORT TEXT 2-"),
        # sg.Button("Save Plot", key="-EXPORT PLOT-")]
    ]

    main_column = [[sg.Text('Plots')],
    [sg.HSep()],
    [sg.HSep()],
    [sg.Column(image_column, key="-FRAME COLUMN-", element_justification='c'), sg.Column(plot_column, key="-PLOT COLUMN-", element_justification='c')],
    [sg.Graph(canvas_size=(graph_size[0]-95, 20), graph_bottom_left=(0, 0), graph_top_right=(graph_size[0]-95, 25), background_color="white", float_values = True, key="-PLOT LEGEND-"), sg.Text("And here", key="-OVER TIME PLOT TITLE-")],
    [sg.Graph(canvas_size=(long_graph_size[0], long_graph_size[1]), graph_bottom_left=(0, 0), graph_top_right=(long_graph_size[0], long_graph_size[1]), change_submits=True, drag_submits=True, background_color="white", float_values = True, key="-OVER TIME PLOT 1-")],
    [sg.Graph(canvas_size=(long_graph_size[0], long_graph_size[1]), graph_bottom_left=(0, 0), graph_top_right=(long_graph_size[0], long_graph_size[1]), change_submits=True, drag_submits=True, background_color="white", float_values = True, key="-OVER TIME PLOT 2-", visible=False)]   
    ]
    right_column = [[sg.Text('Computed metrics:')],
    [sg.HSep()],
    [sg.HSep()],
    [sg.Multiline("", expand_x=True, expand_y=True, disabled=True, size=(20,25), key="-COMPUTED METRICS-")]
    
    ]

    layout = [
        [
            sg.Column(left_column, key="-MAIN OPTIONS COL-", size=(250, 650)),
            sg.VSeperator(),
            sg.Column(main_column, key="-MAIN PLOT DISPLAY-", size=(1000, 650)),
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
        vals_plot_0 = data[key]

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

    dot_size = draw_axes(graph, ax_lims, axes_labels, True)
    #color_select = random.sample(colors, len(data))
    if graph_type == GraphType.LINE_GRAPH: #must be angles over time
        for key in range(len(data)):
            print("")
            print("data shape: (", len(data), ", ", len(data[key]), ", ", len(data[key][0]), ")")
            y_peaks = mm.getPeaks(data[key][:][1], 30)
            if len(y_peaks[0]) > 0:
                print("I found ", len(y_peaks[0]), " weird peaks in angle over time at ", y_peaks)
                for peak in y_peaks[0]:
                    if peak > 0 and peak < len(data[key][:][1])-1:
                        print("peak is ", data[key][peak][1])
                        print("neighbors are ", data[key][peak-1][1], " and ", data[key][peak+1][1])
                        val = (data[key][peak-1][1] + data[key][peak+1][1])/2.0
                        print("so I am replacing ", data[key][peak][1], " with ", val)
                        data[key][peak][1] = val
                y_peaks = mm.getPeaks(data[key][:][1], 30)
                print("And after replacement I found ", len(y_peaks[0]), " weird peaks in the y values at ", y_peaks)

            line_color = colors[key]
            for point in range(len(data[key])-1):
                if data[key][point][2] > mm.GetPlotSpecificInfo("angles over time")[0]:
                    graph.draw_line((data[key][point][0], data[key][point][1]), (data[key][point+1][0], data[key][point+1][1]), width=dot_size, color=line_color)
    elif graph_type == GraphType.POINT_GRAPH: #must be point cloud
        for key in range(len(data)):
            temp_thing = np.array(data[key])
            x_peaks = signal.find_peaks(temp_thing[:,0], threshold=100.0)
            if len(x_peaks[0]) > 0:
                print("")
                print("I found ", len(x_peaks[0]), " weird peaks in the x values of point cloud at ", x_peaks)
                for peak in x_peaks[0]:
                    if peak > 0 and peak < temp_thing.shape[0]-1:
                        print("peak is ", temp_thing[peak,0])
                        print("neighbors are ", temp_thing[peak-1,0], " and ", temp_thing[peak+1,0])
                        val = (temp_thing[peak-1,0] + temp_thing[peak+1,0])/2.0
                        print("so I am replacing ", temp_thing[peak,0], " with ", val)
                        data[key][peak][0] = val
            
            y_peaks = signal.find_peaks(temp_thing[:,1], threshold=100.0)
            if len(y_peaks[0]) > 0:
                print("")
                print("I found ", len(y_peaks[0]), " weird peaks in the y values of point cloud at ", y_peaks)
                for peak in y_peaks[0]:
                    if peak > 0 and peak < temp_thing.shape[1]-1:
                        print("peak at ", peak, " is ", temp_thing[peak,1])
                        print("neighbors are ", temp_thing[peak-1,1], " and ", temp_thing[peak+1,1])
                        val = (temp_thing[peak-1,1] + temp_thing[peak+1,1])/2.0
                        print("so I'm replacing ", temp_thing[peak,1], " with ", val)
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
        max_x_0 = max([point[0] for point in vals_plot_0])
        max_y_0 = max([point[1] for point in vals_plot_0])
        min_x_0 = min([point[0] for point in vals_plot_0])
        min_y_0 = min([point[1] for point in vals_plot_0])
        max_x_1 = max([point[0] for point in vals_plot_1])
        max_y_1 = max([point[1] for point in vals_plot_1])
        min_x_1 = min([point[0] for point in vals_plot_1])
        min_y_1 = min([point[1] for point in vals_plot_1])
        
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

    if graph_type == GraphType.LINE_GRAPH: # must be relative pos over time
        
        for key in range(len(data)):
            plot_data = data[key]
            line_color = colors[key]
            for plot in range(len(plot_data)):
                temp_thing = np.array(plot_data[plot])
                y_peaks = signal.find_peaks(temp_thing[:,1], threshold=30.0)
                if len(y_peaks[0]) > 0:
                    print("I found ", len(y_peaks[0]), " weird peaks in plot #", plot, " of relative pos over time at ", y_peaks)
                    for peak in y_peaks[0]:
                        if peak > 0 and peak < temp_thing.shape[1]-1:
                            print("peak is ", temp_thing[peak][1])
                            print("neighbors are ", temp_thing[peak-1][1], " and ", temp_thing[peak+1][1])
                            val = (temp_thing[peak-1][1] + temp_thing[peak+1][1])/2.0
                            print("replacing ", temp_thing[peak][1], " with ", val)
                            plot_data[plot][peak][1] = val
                    # y_peaks = mm.getPeaks(plot_data[plot][:][1], 100)
                    # print("And after replacement I found ", len(y_peaks[0]), " weird peaks in the y values at ", y_peaks)

                for point in range(len(plot_data[plot])-1):
                    if plot_data[plot][point][2] > mm.GetPlotSpecificInfo("relative position over time")[0]:
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
    frames = [(file_loc+'/video_frames/'+val) for val in file_list if val.lower().endswith((".jpg")) or val.lower().endswith((".png"))]
    frames.sort()
    return frames

def read_pose_files(file_loc):
    try:
        file_list = os.listdir(file_loc+'/pose_info')
    except:
        file_list = []
    
    chosen_files = [val for val in file_list if val.lower().endswith((".json"))]
    chosen_files = sorted(chosen_files)
    return chosen_files

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
        np_data = np.array(data[i])
        temp_total_count, temp_num_count = mm.getAxesCrossedCounts(np_data[:,0], ("right" in name))
        total_track_point_text = total_track_point_text + "\n" + name + " : \n - crossed body midline " + str(temp_num_count) + " times\n - " + str(round(temp_total_count / fps,2)) + " sec spent crossed\n"
        
        temp_total_count, temp_num_count = mm.getAxesCrossedCounts(np_data[:,1], True)
        total_track_point_text = total_track_point_text + " - raised above shoulders " + str(temp_num_count) + " times\n - " + str(round(temp_total_count / fps,2)) + " sec spent raised\n"
        
        data_x_mean = mm.getMean(np_data[:,0])
        data_y_mean = mm.getMean(np_data[:,1])
        data_x_var = mm.getSTD(np_data[:,0])
        data_y_var = mm.getSTD(np_data[:,1])
        total_track_point_text = total_track_point_text + " - average position ( " + str(round(data_x_mean,2)) + ", " + str(round(data_y_mean,2)) + " )\n"
        total_track_point_text = total_track_point_text + " - with std of ( "+ str(round(data_x_var,2)) + ", " + str(round(data_y_var,2)) + " )\n"
        
        filter = np_data[:,2] > mm.GetPlotSpecificInfo("relative position")[0]
        included = sum(filter)
        num_skipped = np_data[:,2].shape[0] - included
        selected = np_data[filter]
        avg_conf = np.mean(selected[:,2])
        total_track_point_text = total_track_point_text + "# frames skipped: "+str(num_skipped) + "\n"
        total_track_point_text = total_track_point_text + "Average confidence: "+str(round(avg_conf,2)) + "\n"

        window['-COMPUTED METRICS-'].update(value=total_track_point_text)
        print(total_track_point_text)

if __name__ == '__main__':
    #matplotlib.use('TkAgg')
    current_layout = get_main_layout()
    window = sg.Window(title="Bilateral Coordination Metric Viewer", layout=current_layout)
    
    #mocap file select variables
    chosen_file = ''
    chosen_files = []

    #mocap video frames
    current_frame = 0
    frames = []
    highlight_frames_iter = []
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

            window["-SELECTED FILE-"].update(value="Currently selected: " + loc_name, visible=True)
            window.TKroot.attributes('-topmost', 1)
            window.TKroot.attributes('-topmost', 0)

        #lets run plotting!
        if event == "-RUN SCRIPT-":
            window['-COMPUTED METRICS-'].Widget.config(wrap='word')
            real_files = [os.path.join(file_loc+'/pose_info', f) for f in chosen_files]
            
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
            highlight_frames_iter.clear()
            highlight_frames_iter = []

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

            highlight = [0 for i in range(max_points)]
            
            if prior_rect[0] == "-FRAME HIGHLIGHT BAR-":
                min_frame = min_x / image_size[0]
                min_frame = int(min_frame*len(frames))
                print("min frame ", min_frame)

                max_frame = max_x / image_size[0]
                max_frame = int(max_frame*len(frames))
                print("max frame ", max_frame)

                highlight[min_frame:max_frame] = [1 for k in range(max_frame-min_frame)]
            elif plot_specfic_data2 is not None and prior_rect[0] == "-OVER TIME PLOT 1-" or prior_rect[0] == "-OVER TIME PLOT 2-":
                for key in range(len(plot_specfic_data2)):
                    for point in range(len(plot_specfic_data2[key])):
                        if highlight[point] == 0 and plot_specfic_data2[key][point][0] > min_x and plot_specfic_data2[key][point][0] < max_x and plot_specfic_data2[key][point][1] > min_y and plot_specfic_data2[key][point][1] < max_y:
                            highlight[point] = 1
            elif plot_specfic_data1 is not None:
                for key in range(len(plot_specfic_data1)):
                    for point in range(len(plot_specfic_data1[key])):
                        if highlight[point] == 0 and plot_specfic_data1[key][point][0] > min_x and plot_specfic_data1[key][point][0] < max_x and plot_specfic_data1[key][point][1] > min_y and plot_specfic_data1[key][point][1] < max_y:
                            highlight[point] = 1
            
            
            highlight_frames_iter = [i for i in range(max_points) if highlight[i]]

            print(f"grabbed rectangle from {start_point} to {end_point}")
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            
            if len(highlight_frames_iter) > 0:
                current_frame = highlight_frames_iter[0]
                window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
                current_frame_mark = display_frame(current_frame)
                window["-SCRUB BAR-"].update(value=current_frame)
                window["-SELECTED FRAMES-"].update(value="Selected keyframe 1 / "+str(len(highlight_frames_iter)))
                
                for mark_i in highlight_marks:
                    window["-FRAME HIGHLIGHT BAR-"].delete_figure(mark_i)
                highlight_marks.clear()
                window["-FRAME HIGHLIGHT BAR-"].Erase()
                for frame_i in highlight_frames_iter:
                    frame_loc = frame_i / len(frames)
                    frame_loc = frame_loc * image_size[0]
                    id = window["-FRAME HIGHLIGHT BAR-"].draw_rectangle((frame_loc, 7), (frame_loc+1, 0), fill_color='red', line_color="red")
                    highlight_marks.append(id)
            else:
                window["-SELECTED FRAMES-"].update(value="No keyframes selected")
            
            
        if event == "-LEFT FRAME-":
            try:
                index = highlight_frames_iter.index(current_frame)
                index = index - 1
            except:
                index = 0

            if index < 0:
                index = len(highlight_frames_iter) - 1
            
            if len(highlight_frames_iter) == 0:
                current_frame = None
            else:
                current_frame = highlight_frames_iter[index]
                window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
                current_frame_mark = display_frame(current_frame)
                window["-SCRUB BAR-"].update(value=current_frame)
                window["-SELECTED FRAMES-"].update(value="Selected keyframe " + str(index + 1)+" / "+str(len(highlight_frames_iter)))

        if event == "-RIGHT FRAME-":
            try:
                index = highlight_frames_iter.index(current_frame)
                index = index + 1
            except:
                index = 0

            if index > len(highlight_frames_iter) - 1:
                index = 0
            
            if len(highlight_frames_iter) == 0:
                current_frame = None
            else:
                current_frame = highlight_frames_iter[index]
                window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
                current_frame_mark = display_frame(current_frame)
                window["-SCRUB BAR-"].update(value=current_frame)
                window["-SELECTED FRAMES-"].update(value="Selected keyframe " + str(index + 1)+" / "+str(len(highlight_frames_iter)))


        #video events
        if event == "-SCRUB BAR-":
            
            current_frame = int(values['-SCRUB BAR-'])
            window["-FRAME HIGHLIGHT BAR-"].delete_figure(current_frame_mark)
            current_frame_mark = display_frame(current_frame)
            window["-SELECTED FRAMES-"].update(value="Not one of the "+str(len(highlight_frames_iter)) + " selected keyframes")

   
    window.close()