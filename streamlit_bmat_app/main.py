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

# full page configs
st.set_page_config(
    page_title="BMAT",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# MediaPipe helper functions
@st.cache_resource
def mediapipe_process(bytesio):

    output = StringIO()
    csv_writer = writer(output)
    csv_writer.writerow(['time', 'frame', 'keypoint_name', 'x', 'y', 'z'])

    if bytesio is None:
        output.seek(0)
        df = pd.read_csv(output)
        return None, df
    

    print("MEDIAPIPE PROCESS")
    if os.path.isfile("./temp.mp4"): 
        print("REMOVING ", "./temp.mp4")
        os.remove("./temp.mp4")   

    #tracked_temp_file_to_save = open("./temp.mp4", "wb") #NamedTemporaryFile(suffix=".mp4", delete=False)

    with NamedTemporaryFile(suffix="mp4") as temp:
        temp.write(bytesio.getbuffer())
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            readcap = cv2.VideoCapture(temp.name)
            fps = int(readcap.get(cv2.CAP_PROP_FPS))
            w = int(readcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(readcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
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
            readcap.release()

    output.seek(0)
    df = pd.read_csv(output)
    
    file = open("./temp.mp4", "rb")
    return file, df


# sidebar
with st.sidebar:
    st.title('BMAT')

    uploaded_vid = st.file_uploader("Choose a video...", type=['mp4','mov', 'avi'])
    
    # process uploaded video
    tracked_file, my_df = mediapipe_process(uploaded_vid)

    # process trackpoints
    trackpoint_choices=["NOSE", "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
    selected_trackpoints = st.multiselect('Select Trackpoints', trackpoint_choices)

    # process graph types
    plot_choices = ['relative position', 'relative angle']
    selected_plot_type = st.selectbox('Select a plot type', plot_choices)

# plotting functions
def make_pointcloud(df):
    df = df.loc[df['keypoint_name'].isin(selected_trackpoints)]
    fig = px.scatter(df, x='x', y='y', title="Position per Frame", color='keypoint_name')

    return fig

def make_overtime(df, vert=True):
    my_title = "Horizontal Position Over Time"
    col = 'x'
    if vert:
        my_title = "Vertical Position Over Time"
        col = 'y'

    fig = px.line(df, x="time", y=col, title=my_title)

    return fig

# layout
col = st.columns((1,1), gap='medium')
with col[0]:
    st.markdown('#### Video')
    if uploaded_vid is not None:
        #video_file = open(old_file, "rb")
        print("DISPLAYING ", tracked_file.name)
        video_bytes = tracked_file.read()
        col[0].video(video_bytes)
        
        if 'video' not in st.session_state:
            st.session_state['video'] = uploaded_vid.name

with col[1]:
    st.markdown('#### Point Cloud')
    point_cloud = make_pointcloud(my_df)
    st.plotly_chart(point_cloud, use_container_width=True)

st.markdown('#### Over Time')