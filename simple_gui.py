from fileinput import filename
import PySimpleGUI as sg
import os.path
import movement_metrics as mm
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

win_size = (800, 600)
window = None

def get_main_layout():
    left_column = [
        [sg.Text("Plot Settings")],
        [sg.HSep()],
        [sg.HSep()],
        [sg.Text('Video to process')],
        [sg.Text(key="-VIDEO FILE-", size=(25,1), enable_events=False, visible=False)], 
        [sg.Button(button_text='Browse', size=(23,1),enable_events=True,key="-VIDEO BUTTON-")], 
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
                "normalized distance from center", "velocity heat map", "speed over time", "velocity over time", 
                "aperature over time"], enable_events=True, size=(25,5), key="-PLOT LIST-"
            )
        ],
        [sg.HSep()],
        [sg.Button(button_text='PLOT', size=(25,1),enable_events=True,key="-RUN SCRIPT-", visible=False)]
    ]
    main_column = [[sg.Text('Plots')],
    [sg.HSep()],
    [sg.HSep()],
    #[sg.Video()], ##TODO
    [sg.Canvas(key="-PLOT CANVAS-")],
    [sg.Text("Export plot as:"), 
        sg.InputText(size=(20,1), key="-PLOT NAME-"), 
        sg.Text(".png"),
        sg.Button("Save Plot", key="-EXPORT PLOT-")]
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
            sg.Column(main_column, key="-MAIN PLOT DISPLAY-", size=(500, 600)),
            sg.VSeperator(),
            sg.Column(right_column, key="-MAIN SCRIPT SETTINGS-", size=(150,600))
        ]
    ]
    return layout

def get_frame_select_layout():
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

    # image_viewer_column = [
    #     [sg.Text("Choose the frame info files from the list on left:"), 
    #         sg.Button("Clear", key="-CLEAR CURRENT-")],
    #     [sg.Listbox(
    #             values=chosen_files, size=(40,18), key="-CHOSEN FILES-"
    #         )],
    #     [sg.Button(button_text='Next',size=(10,2),enable_events=True,key="-NEXT BUTTON-")],
    # ]

    layout = [
        [
            sg.Column(file_list_column),
            #sg.VSeperator(),
            #sg.Column(image_viewer_column),
        ]
    ]

    return layout

def display_file_select(chosen_file):
    sub_window = sg.Window("Video Selection", get_frame_select_layout(), modal=True)
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
                if os.path.isfile(os.path.join(file_loc, f)) and (f.lower().endswith((".mp4")) or f.lower().endswith((".mov"))):
                    file_options.append(f)
                    if f == chosen_file:
                        selected_i.append(i)
            sub_window["-FILE LIST OPTIONS-"].update(values=file_options, set_to_index=selected_i)

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

    
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    current_layout = get_main_layout()
    window = sg.Window(title="Bilateral Coordination Metric Viewer", layout=current_layout)
    
    #mocap file select variables
    chosen_file = ''
    

    #event loop
    file_loc = ""
    while current_layout:
        event, values = window.read()
        
        #overall events
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        #file select events
        if event == "-VIDEO BUTTON-":
            if window["-VIDEO BUTTON-"].get_text() == "Process":
                #run open pose
                #process_video(chosen_file)
                print(chosen_file)
            elif window["-VIDEO BUTTON-"].get_text() == "Browse":
                window.disable()
                file_loc, chosen_file = display_file_select(chosen_file)
                if len(chosen_file) > 0:
                    window["-VIDEO FILE-"].update(value=chosen_file, visible=True)
                    window["-VIDEO BUTTON-"].update("Process")
                window.enable()
        #lets run plotting!
        elif event == "-RUN SCRIPT-":
            #real_files = [os.path.join(file_loc, f) for f in chosen_files]
            
            plot_type = values["-PLOT LIST-"]
            track_points = values["-TRACK POINT LIST-"]
            
            #fig = mm.run_script(real_files, plot_type[0], track_points, 
            #values["-FPS-"], values["-PIX SCALE-"], values["-CONV WIDTH-"])
            #draw_figure(window['-PLOT CANVAS-'].TKCanvas, fig)

        elif event == "-EXPORT PLOT-":
            try:
                matplotlib.pyplot.savefig(values["-PLOT NAME-"] + ".png", bbox_inches='tight')
                window["-PLOT NAME-"].update("")
            except:
                print("ERROR: Couldn't save plot")
        if len(chosen_file) > 0 and len(values["-PLOT LIST-"]) > 0 and len(values["-TRACK POINT LIST-"]) > 0:
            window["-RUN SCRIPT-"].update(visible=True)
        else:
            window["-RUN SCRIPT-"].update(visible=False)
                
    

    window.close()