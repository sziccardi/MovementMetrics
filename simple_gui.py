import PySimpleGUI as sg
import os.path

win_size = (100, 100)
window = None

def get_main_layout():
    left_column = [[sg.Text('Mocap Files')],
    [sg.Button(button_text='Browse',size=(10,1),enable_events=True,key="-BROWSE FILES-"), 
    sg.Listbox(
                values=[], size=(30,8), key="-FILE LIST-", visible=False, enable_events=False
            )], 
    [sg.Text('Track Points')],
    [sg.Listbox(
        values=['head','spine top','shoulder right','elbow right','wrist right','shoulder left',
    'elbow left','wrist left','spine base','hip right','knee right','ankle right','hip left',
    'knee left','ankle left','eye right','eye left','ear right','ear left','big toe left',
    'little toe left','heel left','big toe right','little toe right','heel right','palm left',
    'thumb base left','thumb 1 left','thumb 2 left','thumb tip left','pointer base left',
    'pointer 1 left','pointer 2 left','pointer tip left','middle base left','middle 1 left',
    'middle 2 left','middle tip left','ring base left','ring 1 left','ring 2 left','ring tip left',
    'pinky base left','pinky 1 left','pinky 2 left','pinky tip left','palm right','thumb base right',
    'thumb 1 right','thumb 2 right','thumb tip right','pointer base right','pointer 1 right',
    'pointer 2 right','pointer tip right','middle base right','middle 1 right','middle 2 right',
    'middle tip right','ring base right','ring 1 right','ring 2 right','ring tip right',
    'pinky base right','pinky 1 right','pinky 2 right','pinky tip right'], enable_events=True, size=(30,5), key="-TRACK POINT LIST-", select_mode='multiple'
    )], 
    [sg.Text('Available Plot Types')],
    [ 
        sg.Listbox(
            values=["point cloud", "centroid of motion", "position spectrum", "distance from center", "normalized distance from center", "velocity heat map", "speed over time", "velocity over time", "aperature over time"], enable_events=True, size=(30,5), key="-PLOT LIST-"
        )
    ],
    [sg.Button(button_text='Plot',size=(10,2),enable_events=True,key="-RUN SCRIPT-")]
    ]
    main_column = [[sg.Text('Plots')]]

    layout = [
        [
            sg.Column(left_column, key="-MAIN OPTIONS COL-"),
            sg.VSeperator(),
            sg.Column(main_column, key="-MAIN PLOT DISPLAY-"),
        ]
    ]
    return layout

def get_frame_select_layout():
    file_list_column = [
        [
            sg.Text("File Explorer"), sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse()
        ],
        [ 
            sg.Listbox(
                values=[], enable_events=True, size=(50,20), key="-FILE LIST OPTIONS-",select_mode='multiple'
            )
        ]
    ]

    image_viewer_column = [
        [sg.Text("Choose the frame info files from the list on left:")],
        [sg.Listbox(
                values=[], size=(40,18), key="-CHOSEN FILES-"
            )],
        [sg.Button(button_text='Next',size=(10,2),enable_events=True,key="-NEXT BUTTON-")],
    ]

    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    return layout

def display_file_select():
    sub_window = sg.Window("Mocap Files", get_frame_select_layout(), modal=True)
    chosen_files = []
    while True:
        event, values = sub_window.read()
        if event == "-NEXT BUTTON-":
            break
        elif event == "-BACK BUTTON-":
            chosen_files.clear()
            break
        elif event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".json"))
            ]
            sub_window["-FILE LIST OPTIONS-"].update(fnames)
        elif event == "-FILE LIST OPTIONS-":  # A file was chosen from the listbox
            try:
                chosen_files = values["-FILE LIST OPTIONS-"]
                sub_window["-CHOSEN FILES-"].update(chosen_files)
            except:
                pass
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    sub_window.close()
    return chosen_files


if __name__ == '__main__':
    
    current_layout = get_main_layout()
    window = sg.Window(title="Bilateral Coordination Metric Viewer", layout=current_layout)
    
    #mocap file select variables
    chosen_files = []
    

    #event loop
    while current_layout:
        event, values = window.read()
        
        #overall events
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        #file select events
        if event == "-BROWSE FILES-":
            window.disable()
            chosen_files = display_file_select()
            if len(chosen_files) > 0:
                window["-FILE LIST-"].update(visible=True)
                window["-BROWSE FILES-"].update(visible=False)
                window["-FILE LIST-"].update(chosen_files)
            window.enable()
    

    window.close()