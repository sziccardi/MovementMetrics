from cgitb import enable
from fileinput import filename
import PySimpleGUI as sg
import os.path

from matplotlib.pyplot import plot
import movement_metrics as mm
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import subprocess
from PIL import Image, ImageTk, ImageColor, ImageDraw
import io
import random

graph_size = (475, 475)
image_size = (475, 475)
colors = ["red","coral", "DarkOrange1", "goldenrod", "gold", "green", "aquamarine4", "blue", "purple", "brown", "grey", "black", "cadet blue", "cornflower blue", "DeepPink", "olive drab"]
window = None
processing_gif = b'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'

def get_main_layout():
    left_column = [
        [sg.Text("Plot Settings")],
        [sg.HSep()],
        [sg.HSep()],
        [sg.Text('Currently selected:')],
        [sg.Text(key="-SELECTED FILE-", size=(25,1), enable_events=False, visible=False)], 
        [sg.Button(button_text='Browse New', size=(23,1),enable_events=True,key="-NEW VIDEO BUTTON-"), sg.Image(data=processing_gif, size=(2,2), key="-PROCESSING GIF-", visible=False)], 
        [sg.Button(button_text='Browse Existing', size=(23,1),enable_events=True,key="-EXISTING VIDEO BUTTON-")],
        [sg.HSep()],
        [sg.Text('Track Points')],
        [sg.Listbox(
        values=['head','spine_top','shoulder_right','elbow_right','wrist_right','shoulder_left',
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
        'pinky_base_right','pinky_1_right','pinky_2_right','pinky_tip_right'], enable_events=True, 
            size=(25,5), key="-TRACK POINT LIST-", select_mode='multiple'
        )], 
        [sg.HSep()],
        [sg.Text('Available Plot Types')],
        [ 
            sg.Listbox(
                values=["point cloud", "centroid of motion", "position spectrum", "distance from center", 
                "normalized distance from center", "velocity heat map", "speed over time", "velocity over time"], 
                enable_events=True, size=(25,5), key="-PLOT LIST-"
            )
        ],
        [sg.HSep()],
        [sg.Button(button_text='PLOT', size=(25,1),enable_events=True,key="-RUN SCRIPT-", visible=False)]
    ]

    image_column = [
        [sg.Text("Processed video will show up here", key="-IMAGE TITLE-")],
        [sg.Image(filename="placeholder.png", size=image_size, key='-FRAME IMAGE-')],
        [sg.Slider(range=(0, 100), default_value=0, disable_number_display=True, orientation='horizontal', size=(53,7), key="-SCRUB BAR-", visible=False, enable_events=True)],
        [sg.Button(button_text='Prev Key Frame', enable_events=True,key="-LEFT FRAME-"), sg.Button(button_text='Next Key Frame', enable_events=True,key="-RIGHT FRAME-")],
    ]

    plot_column = [
        [sg.Text("Plotted data will show up here", key="-PLOT TITLE-")],
        [sg.Graph(canvas_size=graph_size, graph_bottom_left=(0, 0), graph_top_right=graph_size, background_color="white", float_values = True, key="-PLOT CANVAS-")],
        [sg.Graph(canvas_size=(graph_size[0], graph_size[1]/2-3), graph_bottom_left=(0, 0), graph_top_right=graph_size, background_color="white", float_values = True, key="-PLOT CANVAS 2-", visible=False)],
        [sg.Text("Export plot as:", key="-EXPORT TEXT 1-"), 
        sg.InputText(size=(20,1), key="-PLOT NAME-"), 
        sg.Text(".png", key="-EXPORT TEXT 2-"),
        sg.Button("Save Plot", key="-EXPORT PLOT-")]
    ]

    main_column = [[sg.Text('Plots')],
    [sg.HSep()],
    [sg.HSep()],
    [sg.Column(image_column, key="-FRAME COLUMN-", element_justification='c'), sg.Column(plot_column, key="-PLOT COLUMN-", element_justification='c')]   
    ]
    right_column = [[sg.Text('Script settings')],
    [sg.HSep()],
    [sg.HSep()],
    [sg.Text("Camera frame rate")],
    [sg.InputText("30", size=(5,1), key="-FPS-"), sg.Text("FPS")],
    [sg.HSep()],
    [sg.Text("Number of Pixels that \nmake up 1 meter")],
    [sg.Text("NOTE: If left blank, units \nwill be displayed in \npixels")],
    [sg.InputText(size=(5,1), key="-PIX SCALE-"), sg.Text("Pixels")],
    [sg.HSep()],
    [sg.Text("Smoothing amount \nor \nConvolution size")],
    [sg.InputText("15", size=(5,1), key="-CONV WIDTH-"), sg.Text("Frames")]]

    layout = [
        [
            sg.Column(left_column, key="-MAIN OPTIONS COL-", size=(200, 600)),
            sg.VSeperator(),
            sg.Column(main_column, key="-MAIN PLOT DISPLAY-", size=(1000, 600)),
            sg.VSeperator(),
            sg.Column(right_column, key="-MAIN SCRIPT SETTINGS-", size=(150,600))
        ]
    ]
    return layout

def get_video_select_layout():
    file_list_column = [
        [sg.Text("Where is the video you wish to process?")],
        [ 
            sg.In(size=(55, 1), enable_events=True, key="-FOLDER-"), 
            sg.FolderBrowse()
            #sg.Button("Select All", size=(10, 1), enable_events=True, key="-SELECT ALL FILES-")
        ],
        [ 
            sg.Listbox(
                values=[], enable_events=True, size=(60,20), key="-FILE LIST OPTIONS-"
            )
        ],
        [sg.Button(button_text='Next',size=(5,1),enable_events=True,key="-NEXT BUTTON-")]
    ]

    layout = [
        [
            sg.Column(file_list_column)
        ]
    ]

    return layout

def get_frame_select_layout():
    file_list_column = [
        [sg.Text("Which folder are the processed files in?")],
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

def display_file_select(chosen_file, existing):
    my_layout = get_frame_select_layout() if existing else get_video_select_layout()
    sub_window = sg.Window("Video Selection", my_layout, modal=True)
    file_loc = ""
    file_options = []
    while True:
        event, values = sub_window.read()
        if event == "-NEXT BUTTON-":
            break
        elif event == "-BACK BUTTON-":
            chosen_file = ''
            break
        elif event == "-FOLDER-":
            file_loc = values["-FOLDER-"]
            try:
                file_list = os.listdir(file_loc)
            except:
                file_list = []
            selected_i = -1
            for i, f in enumerate(file_list):
                if not existing and os.path.isfile(os.path.join(file_loc, f)) and (f.lower().endswith((".mp4")) or f.lower().endswith((".mov"))):
                    file_options.append(f)
                    if f == chosen_file:
                        selected_i.append(i)
                    sub_window["-FILE LIST OPTIONS-"].update(values=file_options, set_to_index=selected_i)
                elif existing:
                    file_options = []
            

        elif event == "-FILE LIST OPTIONS-":  # A file was chosen from the listbox
            try:
                chosen_file = values["-FILE LIST OPTIONS-"][0]
            except:
                pass
        # elif event == "-SELECT ALL FILES-":
        #     try:
        #         all_indices_len = len(file_options)
        #         sub_window["-FILE LIST OPTIONS-"].update(values=file_options, set_to_index=[i for i in range(all_indices_len)])
        #         chosen_files = file_options
        #         sub_window["-CHOSEN FILES-"].update(chosen_files)
        #     except:
        #         print("ERROR: Could not select all")
        #         pass
        # elif event == "-CLEAR CURRENT-":
        #     chosen_files.clear()
        #     sub_window["-CHOSEN FILES-"].update(chosen_files)
        #     sub_window["-FILE LIST OPTIONS-"].update(set_to_index=[])
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    sub_window.close()
    return file_loc, chosen_file

def create_basic_plot(graph, data, data_labels, axes_labels):
    #because it can be ragged...
    y_min = x_min = 1000000000
    y_max = x_max = -1000000000
    for key in range(len(data)):
        np_data = np.array(data[key])
    
        temp_y_min = np.amin(np_data[:, 1])
        temp_y_max = np.amax(np_data[:, 1])
        temp_x_min = np.amin(np_data[:, 0])
        temp_x_max = np.amax(np_data[:, 0])
        
        if temp_y_min < y_min:
            y_min = temp_y_min
        if temp_x_min < x_min:
            x_min = temp_x_min
        if temp_y_max > y_max:
            y_max = temp_y_max
        if temp_x_max > x_max:
            x_max = temp_x_max
    
    x_range = x_max - x_min
    y_range = y_max - y_min

    scaled_y_min = y_min-y_range/7.0
    scaled_x_min = x_min-x_range/7.0
    scaled_y_max = y_max+y_range/7.0
    scaled_x_max = x_max+x_range/7.0
    
    dot_size=x_range/150

    graph.change_coordinates((scaled_x_min, scaled_y_min),(scaled_x_max, scaled_y_max))

    #draw axes
    ax_x_min = x_min if scaled_x_min > 0 else scaled_x_min
    ax_x_max = x_max if scaled_x_max < 0 else scaled_x_max
    ax_y_min = y_min if scaled_y_min > 0 else scaled_y_min
    ax_y_max = y_max if scaled_y_max < 0 else scaled_y_max

    x_ax_label_pos = x_range/2 + ax_x_min if scaled_x_min > 0 else ax_x_min + x_range/75
    y_ax_label_pos = y_range/2 + ax_y_min if scaled_y_min > 0 else ax_y_max - y_range/75
    x_ax_label_anch = "center" if scaled_x_min > 0 else sg.TEXT_LOCATION_TOP_LEFT
    y_ax_label_anch = "center" if scaled_y_min > 0 else sg.TEXT_LOCATION_TOP_RIGHT

    h_tick_len = x_range/60.0
    v_tick_len = y_range/60.0

    graph.draw_line(
        (ax_x_min, max(0, scaled_y_min + y_range/ 10.0)), 
        (ax_x_max, max(0, scaled_y_min + y_range/ 10.0)), color="black", width=dot_size*0.75) #x axis
        
    for x in range(int(ax_x_min), int(ax_x_max), int(x_range/10.0)):
        graph.draw_line((x, max(0, scaled_y_min + y_range/ 10.0)-v_tick_len), (x, max(0, scaled_y_min + y_range/ 10.0)+v_tick_len))  #Draw a scale
        if x != 0:
            graph.draw_text(str(x), (x, max(0, scaled_y_min + y_range/ 10.0)-2.5*v_tick_len), color='black')  #Draw the value of the scale
    
    graph.draw_text(axes_labels[0], (x_ax_label_pos, max(0, scaled_y_min + y_range/ 10.0)-4.75*v_tick_len), text_location = x_ax_label_anch, color="black")

    graph.draw_line(
        (max(0, scaled_x_min + x_range/ 10.0), ax_y_min), 
        (max(0, scaled_x_min + x_range/ 10.0), ax_y_max), color="black", width=dot_size*0.75) #y axis
    
    for y in range(int(ax_y_min), int(ax_y_max), int(y_range/10.0)):
        graph.draw_line((max(0, scaled_x_min + x_range/ 10.0)-h_tick_len, y), (max(0, scaled_x_min + x_range/ 10.0)+h_tick_len, y))
        if y != 0:
            graph.draw_text(str(y), (max(0, scaled_x_min + x_range/ 10.0)-2.5*h_tick_len, y), color='black')
    
    graph.draw_text(axes_labels[1], (max(0, scaled_x_min + x_range/ 10.0)-5.5*h_tick_len, y_ax_label_pos), angle=90, text_location = y_ax_label_anch, color="black")
    
    color_select = random.sample(colors, len(data))
    for key in range(len(data)):
        for point in range(len(data[key])):
            graph.draw_point((data[key][point][0], data[key][point][1]), dot_size, color=color_select[key])

def create_two_plots(graph, data, data_labels, axes_labels, box_n_whisk):
    print("WARNING: This graph is not yet supported")


def process_video(filename):
    #assuming openpose folder is in the movementmetrics folder
    #cd openpose && ./bin/OpenPoseDemo.exe --video ../test_data/UB_B\ Videos/balls_b.mp4 --write_video ../MovementMetrics/output_vid.avi --display 0 --net_resolution 320x176 --write_json ../MovementMetrics/test && cd ..

    cmd = ['/run/myscript', '--arg', 'value']
    p = subprocess.Popen(cmd)
    return p

def read_frame_files(file_loc):
    try:
        file_list = os.listdir(file_loc+'/video_frames')
    except:
        file_list = []
    frames = [(file_loc+'/video_frames/'+val) for val in file_list if val.lower().endswith((".jpg"))]
        
    return frames

def read_pose_files(file_loc):
    try:
        file_list = os.listdir(file_loc+'/pose_info')
    except:
        file_list = []
    
    chosen_files = [val for val in file_list if val.lower().endswith((".json"))]
    return chosen_files

def get_img_data(f, maxsize=image_size, first=False):
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

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
    
    #plotting variables
    graph = window.Element("-PLOT CANVAS-")
    dragging = False
    start_point = end_point = prior_rect = prior_plot = None

    #event loop
    file_loc = ""
    while current_layout:
        event, values = window.read()
        current_process = None
        #overall events
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        #file select events
        if event == "-NEW VIDEO BUTTON-":
            if window["-NEW VIDEO BUTTON-"].get_text() == "Process Current":
                #run open pose
                print("Processing ", chosen_file)
                chosen_files.clear()
                chosen_files = []
                current_process = process_video(chosen_file)
            elif window["-NEW VIDEO BUTTON-"].get_text() == "Browse New":
                file_loc, chosen_file = display_file_select(chosen_file, False)
                loc_name = file_loc[file_loc.rfind('/')+1:]
                if len(chosen_file) > 0:
                    window["-SELECTED FILE-"].update(value=loc_name, visible=True)
                    window["-NEW VIDEO BUTTON-"].update("Process Current")
                    window["-PROCESSING GIF-"].update(visible=True)
                window.TKroot.attributes('-topmost', 1)
                window.TKroot.attributes('-topmost', 0)
        if event == "-EXISTING VIDEO BUTTON-":
            file_loc, chosen_file = display_file_select(chosen_file, True)
            chosen_files.clear()
            frames.clear()
            chosen_files = read_pose_files(file_loc)
            loc_name = file_loc[file_loc.rfind('/')+1:]
            frames = read_frame_files(file_loc)

            window["-SELECTED FILE-"].update(value=loc_name, visible=True)
            window.TKroot.attributes('-topmost', 1)
            window.TKroot.attributes('-topmost', 0)

        #lets run plotting!
        elif event == "-RUN SCRIPT-":
            
            real_files = [os.path.join(file_loc+'/pose_info', f) for f in chosen_files]
            
            plot_type = values["-PLOT LIST-"]
            track_points = values["-TRACK POINT LIST-"]
            
            data, labels, ax_labels = mm.run_script_get_data(real_files, plot_type[0], track_points, 
            values["-FPS-"], values["-PIX SCALE-"], values["-CONV WIDTH-"])
            
            # if prior_plot:
            #     graph.delete_figure(prior_plot)
            # prior_plot = graph.DrawImage(filename="TEMP.png", location=(0, 500))
            graph.Erase()

            window["-PLOT TITLE-"].update(value=plot_type[0])
            print(plot_type[0])
            if plot_type[0] == "point cloud" or plot_type[0] == "centroid of motion" or plot_type[0] == "distance from center" or plot_type[0] == "normalized distance from center" or plot_type[0] == "speed over time":
                window["-PLOT CANVAS-"].set_size(graph_size)
                window["-PLOT CANVAS 2-"].update(visible=False)
                create_basic_plot(graph, data, labels, ax_labels)
            elif plot_type[0] == "position spectrum" or plot_type[0] == "velocity over time":
                window["-PLOT CANVAS-"].set_size((graph_size[0], graph_size[1]/2-3))
                window["-PLOT CANVAS 2-"].update(visible=True)
                create_two_plots(graph, data, labels, ax_labels, plot_type[0] == "position spectrum")
            
            window["-SCRUB BAR-"].update(visible=True, range=(0, len(frames)-1))

            window['-FRAME IMAGE-'].update(data=get_img_data(frames[current_frame], first=True))
            img_name = frames[current_frame][file_loc.rfind('/')+1:]
            window['-IMAGE TITLE-'].update(value=img_name)

        elif event == "-EXPORT PLOT-":
            try:
                matplotlib.pyplot.savefig(values["-PLOT NAME-"] + ".png", bbox_inches='tight')
                window["-PLOT NAME-"].update("")
            except:
                print("ERROR: Couldn't save plot")
        # and ((current_process is not None and current_process.poll() is not None) or (current_process is None))
        if len(chosen_files) > 0 and len(values["-PLOT LIST-"]) > 0 and len(values["-TRACK POINT LIST-"]) > 0:
            window["-RUN SCRIPT-"].update(visible=True)
        else:
            window["-RUN SCRIPT-"].update(visible=False)

        if event == "-SCRUB BAR-":
            window['-FRAME IMAGE-'].update(data=get_img_data(frames[int(values['-SCRUB BAR-'])], first=False))
            current_frame = int(values['-SCRUB BAR-'])
            img_name = frames[current_frame][file_loc.rfind('/')+1:]
            window['-IMAGE TITLE-'].update(value=img_name)

        if event == "-PLOT CANVAS-":
            x, y = values["-PLOT CANVAS-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect:
                window["-PLOT CANVAS-"].delete_figure(prior_rect)
            if None not in (start_point, end_point):
                prior_rect = window["-PLOT CANVAS-"].draw_rectangle(start_point, end_point, line_color='red')
        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            print(f"grabbed rectangle from {start_point} to {end_point}")
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False    


        if current_process is not None and current_process.poll() is None:
            window['-PROCESSING GIF-'].update_animation(processing_gif,  time_between_frames=100)
        elif current_process is not None and current_process.poll() is not None:
            window["-PROCESSING GIF-"].update(visible=False)
            if file_loc:
                file_list = os.listdir(file_loc+'/pose_info')
                for f in file_list:
                    if f.lower().endswith((".json")):
                        chosen_files.append(f)
                current_process = None

                
    

    window.close()