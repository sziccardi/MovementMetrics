import PySimpleGUI as sg
import os.path

win_size = (100, 100)
window = None

def get_main_layout():
    left_column = []
    center_column = []
    right_column = []

def get_frame_select_layout():
    file_list_column = [
        [
            sg.Text("File Explorer"), sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"), sg.FolderBrowse()
        ],
        [ 
            sg.Listbox(
                values=[], enable_events=True, size=(40,20), key="-FILE LIST-",select_mode='multiple'
            )
        ]
    ]

    image_viewer_column = [
        [sg.Text("Choose the frame info files from the list on left:")],
        [sg.Listbox(
                values=[], size=(40,18), key="-CHOSEN FILE LIST-"
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

def update_layout(new_layout):
    window.update(layout=new_layout)


if __name__ == '__main__':
    
    window = sg.Window(title="File viewer", layout=get_main_layout())
    
    #file select variables
    chosen_files = []
    

    #event loop
    while True:
        event, values = window.read()
        
        #overall events
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        #file select events
        if event == "-NEXT BUTTON-":
            update_layout(get_main_layout())
        elif event == "-BACK BUTTON-":
            chosen_files.clear()
            update_layout(get_main_layout())
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
            window["-FILE LIST-"].update(fnames)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                selected = values["-FILE LIST-"]            
                window["-CHOSEN FILE LIST-"].update(selected)
            except:
                pass

    window.close()