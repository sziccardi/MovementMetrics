import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import os
from tempfile import NamedTemporaryFile
from io import StringIO
from csv import writer 
from streamlit_scatterplot_selection import st_scatterplot
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from streamlit_bokeh3_events import streamlit_bokeh3_events

vidcap = None

# full page configs
st.set_page_config(
    page_title="BMAT",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

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
            
            writecap = cv2.VideoWriter("./temp.mp4", 0x00000021, fps, (w, h)) 
            
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
                writecap.write(image)

                # Write the landmark positions to df
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    new_row = [float(frame_num)/float(fps), frame_num, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z]
                    #df = pd.concat([df, new_row], axis = 0, ignore_index = True)
                    csv_writer.writerow(new_row)

                frame_num += 1

            readcap.release()
            writecap.release()

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
    wrotecap = cv2.VideoCapture("./temp.mp4")
    return df, wrotecap


# sidebar
with st.sidebar:
    st.title('BMAT')

    uploaded_vid = st.file_uploader("Choose a video...", type=['mp4','mov', 'avi'])
    
    # process uploaded video
    my_df, vidcap = mediapipe_process(uploaded_vid)

    # process trackpoints
    trackpoint_choices=["NOSE", "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
    selected_trackpoints = st.multiselect('Select Trackpoints', trackpoint_choices)

    # process graph types
    plot_choices = ['relative position', 'relative angle']
    selected_plot_type = st.selectbox('Select a plot type', plot_choices)

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
def make_pointcloud(col_data):
    # fig = px.scatter(df, x='x', y='y', title="Position per Frame", color='keypoint_name')
    # fig.update_xaxes(fixedrange=True)
    # fig.update_yaxes(fixedrange=True)
    print(col_data.column_names)
    fig = figure(tools="lasso_select,reset", x_axis_label="x pos (px)", y_axis_label="y pos (px)")

    fig.scatter(x='x', y='y', size=2, source=col_data)
    
    col_data.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=col_data),
            code="""
            document.dispatchEvent(
                new CustomEvent("PointCloudSelectEvent", {detail: {indices: cb_obj.indices}})
            )
        """,
        ),
    )
    
    return fig

def make_overtime(df, vert=True):
    my_title = "Horizontal Position Over Time"
    col = 'x'
    if vert:
        my_title = "Vertical Position Over Time"
        col = 'y'
    fig = px.line(df, x="time", y=col, title=my_title, color='keypoint_name')
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig

# layout
col = st.columns((1,1), gap='medium')

    
subset_df = my_df.loc[my_df['keypoint_name'].isin(selected_trackpoints)]
selected_df = subset_df.copy()
with col[1]:
    st.markdown('#### Point Cloud')
    
    col_data = ColumnDataSource(subset_df)

    #point_cloud = make_pointcloud(col_data)
    point_cloud = figure(tools="lasso_select,reset", x_axis_label="x pos (px)", y_axis_label="y pos (px)")

    point_cloud.scatter(x='x', y='y', size=2, source=col_data)
    
    col_data.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=col_data),
            code="""
            document.dispatchEvent(
                new CustomEvent("PointCloudSelectEvent", {detail: {indices: cb_obj.indices}})
            )
        """,
        ),
    )

    event_result = streamlit_bokeh3_events(
        events="PointCloudSelectEvent",
        bokeh_plot=point_cloud,
        key="point_cloud",
        debounce_time=100,
        refresh_on_update=True
    )
    #st.bokeh_chart(point_cloud)

    
    #st.bokeh_chart(point_cloud, use_container_width=True)

    # some event was thrown
    if event_result is not None:
        # PointCloudSelectEvent was thrown
        if "PointCloudSelectEvent" in event_result:
            indices = event_result["PointCloudSelectEvent"].get("indices", [])
            selected_df = subset_df.iloc[indices]
            st.session_state['selected_data_indices'] = indices
            st.session_state['frame_num'] = min(selected_df['frame'])
            

with col[0]:
    st.markdown('#### Video')
    if uploaded_vid is not None:
        
        # video_bytes = tracked_file.read()
        # col[0].video(video_bytes)
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


st.markdown('#### Horizontal Position Over Time')
over_time_horiz = make_overtime(subset_df, False)
#st.line_chart(subset_df, x='time', y='x', color='keypoint_name')
st.plotly_chart(over_time_horiz, use_container_width=True)
over_time_vert = make_overtime(subset_df, True)
st.plotly_chart(over_time_vert, use_container_width=True)
#st.markdown('#### Vertical Position Over Time')
#st.line_chart(subset_df, x='time', y='y', color='keypoint_name')