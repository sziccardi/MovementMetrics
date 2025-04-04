import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
#import plotly.express as px
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import os
from tempfile import NamedTemporaryFile
from io import StringIO
from csv import writer 
#from streamlit_scatterplot_selection import st_scatterplot
from streamlit_dimensions import st_dimensions
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, CDSView, IndexFilter
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
from bokeh.io import curdoc
from bokeh.layouts import column
from streamlit_bokeh3_events import streamlit_bokeh3_events

vidcap = None
my_df = None

# MediaPipe helper functions
@st.cache_resource
def mediapipe_process(bytes_to_load):
    
    output = StringIO()
    csv_writer = writer(output)
    csv_writer.writerow(['time', 'frame', 'keypoint_name', 'x', 'y', 'z'])

    if bytes_to_load is None:
        output.seek(0)
        df = pd.read_csv(output)
        return df, None
    

    print("MEDIAPIPE PROCESS")
    if os.path.isfile("./temp.mp4"): 
        print("REMOVING ", "./temp.mp4")
        os.remove("./temp.mp4")   

    #tracked_temp_file_to_save = open("./temp.mp4", "wb") #NamedTemporaryFile(suffix=".mp4", delete=False)
    readcap = None
    with NamedTemporaryFile(suffix="mp4") as temp:
        temp.write(bytes_to_load.getbuffer())
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            readcap = cv2.VideoCapture(temp.name)
            fps = int(readcap.get(cv2.CAP_PROP_FPS))
            w = int(readcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(readcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #num_frames = int(readcap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            #writecap = cv2.VideoWriter("./temp.mp4", 0x00000021, fps, (w, h)) 
            
            frame_num = 0
            while readcap.isOpened():
                # Read frame from video
                success, image = readcap.read()
                if not success:
                    print("Ignoring empty camera frame at frame #", frame_num)
                    break

                # Process frame with mediapipe
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Save the annotated image in the temp tracked video
                #writecap.write(image)

                # Write the landmark positions to df
                if results.pose_landmarks is not None:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        new_row = [float(frame_num)/float(fps), frame_num, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z]
                        #df = pd.concat([df, new_row], axis = 0, ignore_index = True)
                        csv_writer.writerow(new_row)

                frame_num += 1

            readcap.release()
            #writecap.release()
    
    output.seek(0)
    df = pd.read_csv(output)
    thing1 = df.loc[df['keypoint_name'] == "LEFT_SHOULDER"]
    thing1 = thing1.set_axis([f for f in thing1['frame']])
    thing1 = thing1[['x', 'y', 'z']]

    thing2 = df.loc[df['keypoint_name'] == "RIGHT_SHOULDER"]
    thing2 = thing2.set_axis([f for f in thing2['frame']])
    thing2 = thing2[['x', 'y', 'z']]

    sternum = thing1.add(thing2) / 2.0
    
    keypoints = df['keypoint_name'].unique()
    
    for key in keypoints:
        keything = df.loc[df['keypoint_name'] == key, ['x', 'y', 'z']]
        keysternum = sternum.set_axis([f for f in keything.index])

        df.loc[df['keypoint_name'] == key,['x', 'y', 'z']] = (keything - keysternum)
        df.loc[df['keypoint_name'] == key,'y'] = df.loc[df['keypoint_name'] == key,'y'] * -1.0
    #file = open("./temp.mp4", "rb")
    
    st.session_state['frame_num'] = 0
    #wrotecap = cv2.VideoCapture("./temp.mp4")
    return df


# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file or folder', filenames)
#     return os.path.join(folder_path, selected_filename)


def get_frame(frame_num, cap):
    #cap = cv2.VideoCapture("./temp.mp4")
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if frame_num is not None and frame_num >= 0 and frame_num <= totalFrames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if (ret):
            return frame
        else:
            print("Failed load from cap")
    return None

# plotting functions
def make_pointcloud(col_data, col_width):
    # fig = px.scatter(df, x='x', y='y', title="Position per Frame", color='keypoint_name')
    # fig.update_xaxes(fixedrange=True)
    # fig.update_yaxes(fixedrange=True)
    keys = np.unique(col_data.data["keypoint_name"])
    num = len(keys)
    pal = Category10[max(3, num)]
    pal = pal[:num]
    
    index_cmap = factor_cmap('keypoint_name', palette=pal, 
                         factors=np.unique(col_data.data["keypoint_name"]))
    
    new_x_min = new_x_max = new_y_min = new_y_max = 0
    if len(col_data.data['x']) > 0: 
        x_width = np.max(col_data.data['x']) - np.min(col_data.data['x'])
        x_mid = np.min(col_data.data['x']) + x_width/2.0
        new_x_width = x_width*1.1
        new_x_min = x_mid - new_x_width / 2.0 
        new_x_max = x_mid + new_x_width / 2.0 
        y_width = np.max(col_data.data['y']) - np.min(col_data.data['y'])
        y_mid = np.min(col_data.data['y']) + y_width/2.0
        new_y_width = y_width*1.1
        new_y_min = y_mid - new_y_width / 2.0 
        new_y_max = y_mid + new_y_width / 2.0 
    fig = figure(tools="lasso_select,reset", width=col_width, x_axis_label="x pos (px)", y_axis_label="y pos (px)", x_range=(new_x_min, new_x_max), y_range=(new_y_min, new_y_max))
    
    fig.line(x=[0, 0], y=[fig.y_range.start, fig.y_range.end], line_width=3, color='black')
    fig.line(x=[fig.x_range.start, fig.x_range.end], y=[0, 0], line_width=3, color='black')
    
    if 'selected_data_indices' in st.session_state:
        print("FOUND A SELECTION, DRAWING SELECTION POINTCLOUD")
        fig.scatter(x='x', y='y', size=2, source=col_data,fill_alpha=0.2, line_alpha=0.2, color=index_cmap, legend_group='keypoint_name')
        view = CDSView(source=col_data, filters = [IndexFilter(st.session_state['selected_data_indices'])])
        fig.scatter(x='x', y='y', size=4, source=col_data, view=view, fill_alpha=1.0, color=index_cmap, legend_group='keypoint_name')
    else: 
        fig.scatter(x='x', y='y', size=2, source=col_data,fill_alpha=0.6, color=index_cmap, legend_group='keypoint_name')

    col_data.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=col_data),
            code="""
            document.dispatchEvent(
                new CustomEvent("PointCloudSelectEvent", {detail: {indices: source.selected.indices}})
            )
        """,
        ),
    )
    
    
    return fig

def make_overtime(col_data, col_width):
        
    keys = np.unique(col_data.data["keypoint_name"])
    
    num = len(keys)
    pal = Category10[max(3, num)]
    pal = pal[:num]
    
    new_t_min = new_t_max = new_x_min = new_x_max = new_y_min = new_y_max = 0
    if len(col_data.data['time']) > 0: 
        t_width = np.max(col_data.data['time']) - np.min(col_data.data['time'])
        t_mid = np.min(col_data.data['time']) + t_width/2.0
        new_t_width = t_width*1.1
        new_t_min = t_mid - new_t_width / 2.0 
        new_t_max = t_mid + new_t_width / 2.0 

        x_width = np.max(col_data.data['x']) - np.min(col_data.data['x'])
        x_mid = np.min(col_data.data['x']) + x_width/2.0
        new_x_width = x_width*1.1
        new_x_min = x_mid - new_x_width / 2.0 
        new_x_max = x_mid + new_x_width / 2.0 

        y_width = np.max(col_data.data['y']) - np.min(col_data.data['y'])
        y_mid = np.min(col_data.data['y']) + y_width/2.0
        new_y_width = y_width*1.1
        new_y_min = y_mid - new_y_width / 2.0 
        new_y_max = y_mid + new_y_width / 2.0 
    
    
    
    x_subset = None
    y_subset = None
    t_subset = None
    key_subset = None
    if 'selected_data_indices' in st.session_state:
        print("FOUND A SELECTION, DRAWING SELECTION OVER TIME")
        x_subset = col_data.data['x'][st.session_state['selected_data_indices']]
        y_subset = col_data.data['y'][st.session_state['selected_data_indices']]
        t_subset = col_data.data['time'][st.session_state['selected_data_indices']]
        key_subset = col_data.data['keypoint_name'][st.session_state['selected_data_indices']]
    
    all_t_data=[]
    all_x_data=[]
    all_y_data=[]  
    selected_t_data=[]
    selected_x_data=[]
    selected_y_data=[]
    for i,key in enumerate(keys):
        key_data = col_data.data['keypoint_name']
        inds = np.where(key_data == key)[0]
        x_data = col_data.data['x'][inds]
        all_x_data.append(x_data)
        y_data = col_data.data['y'][inds]
        all_y_data.append(y_data)
        t_data = col_data.data['time'][inds]
        all_t_data.append(t_data)

        if x_subset is not None:
            s_inds = np.where(key_subset == key)[0]
            s_x_data = x_subset[s_inds]
            selected_x_data.append(s_x_data)
            s_y_data = y_subset[s_inds]
            selected_y_data.append(s_y_data)
            s_t_data = t_subset[s_inds]
            selected_t_data.append(s_t_data)

    
    data_dict = {'ts':all_t_data, 'xs':all_x_data, 'ys':all_y_data, 'colors':pal, 'labels':keys}
    selected_data_dict = {'ts':selected_t_data, 'xs':selected_x_data, 'ys':selected_y_data, 'colors':pal, 'labels':keys}
    horiz_cds = ColumnDataSource(data_dict)
    selected_horiz_cds = ColumnDataSource(selected_data_dict)


    vert_cds = ColumnDataSource(data_dict)
    selected_vert_cds = ColumnDataSource(selected_data_dict)

    figh = figure(tools="lasso_select,reset", width=col_width, height=int(col_width/4.0),title="Horizontal Position Over Time", x_axis_label="Time (s)", y_axis_label="Pos (px)", x_range=(new_t_min, new_t_max), y_range=(new_x_min, new_x_max))

    
    if len(selected_t_data) > 0:
        figh.multi_line(xs='ts', ys='xs', line_width=2, line_alpha=0.2, color='colors', legend_field='labels', source=horiz_cds)
        figh.multi_line(xs='ts', ys='xs', line_width=8, line_alpha=0.6, color='colors', legend_field='labels', source=selected_horiz_cds)
    else: 
        figh.multi_line(xs='ts', ys='xs', line_width=2, line_alpha=0.6, color='colors', legend_field='labels', source=horiz_cds)
        figh.scatter(x='ts', y='xs', size=8, source=horiz_cds, fill_alpha=0.6, color='colors')

    
    figh.line(x=[0, 0], y=[figh.x_range.start, figh.x_range.end], line_width=3, color='black')
    figh.line(x=[figh.y_range.start, figh.y_range.end], y=[0, 0], line_width=3, color='black')

    horiz_cds.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=horiz_cds),
            code="""
            document.dispatchEvent(
                new CustomEvent("HorizOverTimeSelectEvent", {detail: {indices: cb_obj.indices}})
            )
        """,
        ),
    )

    figv = figure(tools="lasso_select,reset", width=col_width, height=int(col_width/4.0),title="Vertical Position Over Time", x_axis_label="Time (s)", y_axis_label="Pos (py)", x_range=(new_t_min, new_t_max), y_range=(new_y_min, new_y_max))

    if len(selected_t_data) > 0:
        figv.multi_line(xs='ts', ys='ys', line_width=2, line_alpha=0.2, color='colors', legend_field='labels', source=vert_cds)
        figv.multi_line(xs='ts', ys='ys', line_width=8, line_alpha=0.6, color='colors', legend_field='labels', source=selected_vert_cds)
    else: 
        figv.multi_line(xs='ts', ys='ys', line_width=2, line_alpha=0.6, color='colors', legend_field='labels', source=vert_cds)
    
    figv.line(x=[0, 0], y=[figv.x_range.start, figv.x_range.end], line_width=3, color='black')
    figv.line(x=[figv.y_range.start, figv.y_range.end], y=[0, 0], line_width=3, color='black')

    vert_cds.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=vert_cds),
            code="""
            document.dispatchEvent(
                new CustomEvent("VertOverTimeSelectEvent", {detail: {indices: cb_obj.indices}})
            )
        """,
        ),
    )
    
    p = column(children=[figh, figv], sizing_mode="scale_both")

    return p

def get_metrics(df, plot_type):
    metrics_text = ""
    if "pos" in plot_type:
        for key in df['keypoint_name'].unique():
            y_key_df = df.loc[(df['keypoint_name'] == key) & df['y'] != 0.0].reset_index()
            y_up = y_key_df['y'].gt(0).astype(int)
            
            should_cross = str(sum(y_up[y_up.ne(y_up.shift(1))]))
            should_cross_time = str(sum(y_up))

            midline_cross = midline_cross_time = "N/A"
            x_key_df = df.loc[(df['keypoint_name'] == key) & df['x'] != 0.0].reset_index()
            if 'RIGHT' in key:
                #neg to pos
                x_up = x_key_df['x'].gt(0).astype(int)
                midline_cross = str(sum(x_up[x_up.ne(x_up.shift(1))]))
                midline_cross_time = str(sum(x_up))
            elif 'LEFT' in key:
                #pos to neg
                low = x_key_df['x'].lt(0).astype(int)
                midline_cross = str(sum(low[low.ne(low.shift(1))]))
                midline_cross_time = str(sum(low))

            metrics_text += key + "  \n - Above shoulder:  \n     - Count = " + should_cross + "  \n      - Time spent = " + should_cross_time + " frames    \n - Midline crosses:   \n" + "     - Count = " + midline_cross + "  \n     - Time spent = " + midline_cross_time + " frames    \n  \n"


    return metrics_text


def run():
    print("RERUN")
    global my_df
    global vidcap
    # full page configs
    st.set_page_config(
        page_title="BMAT",
        layout="wide",
        initial_sidebar_state="expanded")

    alt.themes.enable("dark")
    curdoc().theme = "dark_minimal"

    # sidebar
    with st.sidebar:
        st.title('BMAT')

        uploaded_vid = st.file_uploader("Choose a video... [required]", type=['.mp4', '.MP4', '.mov', '.MOV'])
        uploaded_csv = st.file_uploader("Choose a pre-processed file... [optional]", type=['.csv'])
        
        
        
        # process trackpoints
        trackpoint_choices=["NOSE", "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
        selected_trackpoints = st.multiselect('Select Trackpoints', trackpoint_choices)
        #st.session_state['clear_selection'] = True
        if 'selected_trackpoints' in st.session_state:
            same = True
            for point in st.session_state['selected_trackpoints']:
                if point not in selected_trackpoints:
                    same = False
                    break
            
            #st.session_state['clear_selection'] = not same
            if not same and 'selected_data_indices' in st.session_state:
                st.session_state.pop('selected_data_indices')
                #del st.session_state['selected_data_indices']
                print('selected_data_indices' in st.session_state)
                print("DELETING SELECTION")

        st.session_state['selected_trackpoints'] = selected_trackpoints

        # process graph types
        plot_choices = ['relative position', 'relative angle']
        selected_plot_type = st.selectbox('Select a plot type', plot_choices)

        if st.button('Process!'):
            if uploaded_csv is None and uploaded_vid is not None:
                # process uploaded video
                my_df = mediapipe_process(uploaded_vid)
                st.session_state['df'] = my_df

            elif uploaded_csv is not None and uploaded_vid is not None:
                my_df = pd.read_csv(uploaded_csv)
                
                #csv_writer.writerow(['time', 'frame', 'keypoint_name', 'x', 'y', 'z'])
            with NamedTemporaryFile(suffix="mp4") as temp:
                temp.write(uploaded_vid.getbuffer())
                vidcap = cv2.VideoCapture(temp.name)
                st.session_state['frame_num'] = 0
        
        if 'frame_num' in st.session_state: 
            if uploaded_vid is not None:
                with NamedTemporaryFile(suffix="mp4") as temp:
                    temp.write(uploaded_vid.getbuffer())
                    vidcap = cv2.VideoCapture(temp.name)

        if my_df is None and uploaded_csv is not None and uploaded_vid is not None:
            my_df = pd.read_csv(uploaded_csv)

        if my_df is not None:
            filename = ''
            if uploaded_vid is not None:
                i = uploaded_vid.name.rfind('.')
                filename = 'BMAT_'+uploaded_vid.name[:i]+'.csv'
            elif uploaded_csv is not None:
                filename = uploaded_csv.name
            st.download_button('Download pose data', my_df.to_csv().encode("utf-8"), mime="text/csv", file_name=filename)
        elif 'df' in st.session_state:
            my_df = st.session_state['df']
        #st.download_button('Download video data', vidcap., mime="video/mp4", file_name='BMAT_'+filename+".csv")

    # layout
    col = st.columns((1,1), gap='medium')

    subset_df = pd.DataFrame(columns=['time', 'frame', 'keypoint_name', 'x', 'y', 'z'])
    if my_df is not None:
        print("selected track points????")
        subset_df = my_df.loc[my_df['keypoint_name'].isin(selected_trackpoints)]
            

    with col[0]:
        st.markdown('#### Video')
        if uploaded_vid is not None:
            
            if 'frame_num' in st.session_state and int(st.session_state['frame_num']) >= 0 and vidcap is not None:
                num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                t = st.slider("Frame:", value=int(st.session_state['frame_num']), min_value=0, max_value=int(num_frames), step=1)

                st.session_state['frame_num'] = t
                img = get_frame(int(st.session_state['frame_num']), vidcap)
                if img is not None:
                    col[0].image(img)
                
            else:
                col[0].image('TEMP.jpg')
            
            if 'video' not in st.session_state:
                st.session_state['video'] = uploaded_vid.name
        else:
            st.session_state['frame_num'] = -1
            col[0].image('TEMP.jpg')
            # if os.path.isfile("./temp.mp4"): 
            #     print("REMOVING ", "./temp.mp4")
            #     if vidcap is not None:
            #         vidcap.release()
            #     os.remove("./temp.mp4")
    if vidcap is not None:
        vidcap.release()
        vidcap = None
    
    with col[1]:
        st.markdown('#### Metrics')
        with st.container(height=500):
            text = get_metrics(subset_df, selected_plot_type)
            st.markdown(text)

    st.markdown('#### Point Cloud')
        
    selected_df = subset_df.copy()

    col_data_pc = ColumnDataSource(subset_df)
    col_data_ot = ColumnDataSource(subset_df)
    # col_data_otv = ColumnDataSource(subset_df)

    size = st_dimensions(key="main")
    if size is None:
        size={'width':300}    

    print("CALLING MAKE POINT CLOUD")
    point_cloud = make_pointcloud(col_data_pc, int(size['width']))

    pc_event_result = streamlit_bokeh3_events(
        events="PointCloudSelectEvent",
        bokeh_plot=point_cloud,
        key="point_cloud",
        debounce_time=0,
        refresh_on_update=True
    )

    # some event was thrown
    if pc_event_result is not None:
        # PointCloudSelectEvent was thrown
        if "PointCloudSelectEvent" in pc_event_result:
            indices = pc_event_result["PointCloudSelectEvent"].get("indices", [])
            print("POINT CLOUD SELECT EVENT")
            print(indices)
            selected_df = subset_df.iloc[indices]
            st.session_state['selected_data_indices'] = indices
            st.session_state['frame_num'] = min(selected_df['frame'])
            

    st.markdown('#### Position Over Time')
    print("CALLING MAKE OVER TIME")
    over_time = make_overtime(col_data_ot, int(size['width']))
    #st.line_chart(subset_df, x='time', y='x', color='keypoint_name')
    #st.plotly_chart(over_time_horiz, use_container_width=True)
    ot_event_result = streamlit_bokeh3_events(
        events="HorizOverTimeSelectEvent, VertOverTimeSelectEvent",
        bokeh_plot=over_time,
        key="over_time",
        debounce_time=100,
        refresh_on_update=True
    )
    if ot_event_result is not None:
        # PointCloudSelectEvent was thrown
        if "HorizOverTimeSelectEvent" in ot_event_result:
            indices = pc_event_result["HorizOverTimeSelectEvent"].get("indices", [])
            selected_df = subset_df.iloc[indices]
            st.session_state['selected_data_indices'] = indices
            st.session_state['frame_num'] = min(selected_df['frame'])
            print("GRABBED DATA FROM HORIZ")
        elif "VertOverTimeSelectEvent" in ot_event_result:
            indices = pc_event_result["HorizOverTimeSelectEvent"].get("indices", [])
            selected_df = subset_df.iloc[indices]
            st.session_state['selected_data_indices'] = indices
            st.session_state['frame_num'] = min(selected_df['frame'])
            print("GRABBED DATA FROM VERT")


    # st.markdown('#### Vertical Position Over Time')
    # over_time_vert = make_overtime(col_data_otv, int(size['width']), True)
    # #st.plotly_chart(over_time_vert, use_container_width=True)
    # #st.markdown('#### Vertical Position Over Time')
    # #st.line_chart(subset_df, x='time', y='y', color='keypoint_name')

    # otv_event_result = streamlit_bokeh3_events(
    #     events="OverTimeVertSelectEvent",
    #     bokeh_plot=over_time_vert,
    #     key="over_time_vert",
    #     debounce_time=100,
    #     refresh_on_update=True
    # )

    # # some event was thrown
    # if otv_event_result is not None:
    #     # PointCloudSelectEvent was thrown
    #     if "OverTimeHorizSelectEvent" in pc_event_result:
    #         indices = pc_event_result["OverTimeHorizSelectEvent"].get("indices", [])
    #         selected_df = subset_df.iloc[indices]
    #         st.session_state['selected_data_indices'] = indices
    #         st.session_state['frame_num'] = min(selected_df['frame'])

        # st.markdown('''
        # <style>
        # [data-testid="stMarkdownContainer"] ul{
        #     padding-left:40px;
        # }
        # </style>
        # ''', unsafe_allow_html=True)



if __name__ == "__main__":
    run()