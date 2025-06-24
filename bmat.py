
import altair as alt
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
import os
from csv import writer
from io import StringIO
import cv2
import time

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import streamlit as st
from streamlit_bokeh3_events import streamlit_bokeh3_events
from bokeh.io import curdoc
from streamlit_bokeh import streamlit_bokeh
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool, CustomJS

# Globals
video_time = 0
trackpoint_choices = ["NOSE", "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]

color_palette = Category10[len(trackpoint_choices)]
color_palette = color_palette[:len(trackpoint_choices)]

color_palette = dict(zip(trackpoint_choices, color_palette))


def on_video_change():
    if "video_capture" in st.session_state:
        print("releasing old cap")
        st.session_state["video_capture"].release()
        del st.session_state["video_capture"]
    if "video_time" in st.session_state:
        st.session_state["video_time"] = 0
    # TODO: reset skeleton data
    st.session_state["pose_trajectory"] = pd.DataFrame(columns=['frame', 'keypoint_name', 'x', 'y', 'z', 'color', 'prev_x', 'prev_y', 'speed_xy'])
    st.session_state["uploader_key"] += 1
    #st.rerun()
    # uploaded_file = st.session_state["my_uploader"]
    # if uploaded_file is not None:
        # st.write(f"File uploaded: {uploaded_file.name}")


def process_video(video, progress_bar):
    if "video_capture" in st.session_state: #and "pose_trajectory" not in st.session_state
        output = StringIO()
        csv_writer = writer(output)
        csv_writer.writerow(['frame_sec', 'frame', 'keypoint_name', 'x', 'y', 'z', 'color', 'prev_x', 'prev_y', 'speed_xy'])

        if video is None:
            output.seek(0)
            df = pd.read_csv(output)
            return df, None
        
        readcap = st.session_state["video_capture"]
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            fps = int(readcap.get(cv2.CAP_PROP_FPS))
            w = int(readcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(readcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(readcap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_num = 0
            while readcap.isOpened():
                if frame_num%fps == 0:
                    print("PROCESSING FRAME #", frame_num)
                    
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
                progress_bar.progress(frame_num / num_frames)


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
        print("DEBUG")
        print(df['keypoint_name'])
        print(keypoints)
        print(sternum)
        for key in keypoints:
            print(key)
            keything = df.loc[df['keypoint_name'] == key, ['x', 'y', 'z']]
            keysternum = sternum.set_axis([f for f in keything.index])

            df.loc[df['keypoint_name'] == key,['x', 'y', 'z']] = (keything - keysternum)
            df.loc[df['keypoint_name'] == key,'y'] = df.loc[df['keypoint_name'] == key,'y'] * -1.0
        
        keys = df['keypoint_name'].unique()
        for k in keys:
            if k not in color_palette:
                color_palette[k] = '#000000'
        df['color'] = df['keypoint_name'].replace(color_palette)

        #'frame', 'keypoint_name', 'x', 'y', 'z', 'color', 'prev_x', 'prev_y', 'speed_xy'
        df['prev_x'].iloc[1:] = df['x'].iloc[:-1]
        df['prev_y'].iloc[1:] = df['y'].iloc[:-1]

        dx = df['x'].iloc[1:] - df['prev_x'].iloc[1:]
        dy = df['y'].iloc[1:] - df['prev_y'].iloc[1:]
        df['speed_xy'].iloc[1:] = np.sqrt(dx*dx + dy*dy) * fps

        progress_bar.progress(1.0)

        return df
        #     st.session_state["pose_trajectory"] = None
        # for i in range(100):
        #     time.sleep(1 / 200)  # Wait 1/200th of a second
        #     progress_bar.progress(i + 1)  # Update progress bar
    else:
        return pd.DataFrame(columns=['frame', 'keypoint_name', 'x', 'y', 'z', 'color', 'prev_x', 'prev_y', 'speed_xy'])


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


def run_app():
    global trackpoint_choices
    print("RERUN")
    if "video_time" not in st.session_state:
        st.session_state["video_time"] = 0

    st.set_page_config(
        page_title="PyMAT",
        layout="wide",
        initial_sidebar_state="expanded")

    alt.themes.enable("dark")
    curdoc().theme = "dark_minimal"

    csv_reference = None
    video_reference = None

    if "pose_trajectory" not in st.session_state:
        # initialize empty 
        st.session_state["pose_trajectory"] = pd.DataFrame(columns=['frame', 'keypoint_name', 'x', 'y', 'z', 'color', 'prev_x', 'prev_y', 'speed_xy'])
        
        

    with st.sidebar:
        st.title('BMAT')
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 1
        video_reference = st.file_uploader("Choose a video... [required]", on_change=on_video_change, type=['.mp4', '.MP4', '.mov', '.MOV'])
        csv_reference = st.file_uploader("Choose a pre-processed file... [optional]", type=['.csv'], key=st.session_state["uploader_key"])

        # process trackpoints
        selected_trackpoints = st.multiselect('Select Trackpoints', trackpoint_choices)
        st.session_state['selected_trackpoints'] = selected_trackpoints

    
        # if "selected_trackpoints" in st.session_state:
        #     print(st.session_state['selected_trackpoints'])
        # breakpoint()
        # print("video reference:\n", list(video_reference.keys()))

        # process graph types
        plot_choices = ['relative position', 'relative angle']
        selected_plot_type = st.selectbox('Select a plot type', plot_choices)

        if len(st.session_state["pose_trajectory"]) > 0:
            i = video_reference.name.rfind('.')
            filename = 'BMAT_'+video_reference.name[:i]+'.csv'
            st.download_button('Download pose data', st.session_state["pose_trajectory"].to_csv().encode("utf-8"), mime="text/csv", file_name=filename)
        else:
            if csv_reference is None:
                if st.button('Process Trackpoints!'):
                    print("processing...")
                    progress_bar = st.progress(0)
                    df = process_video(video_reference, progress_bar)
                    st.session_state["pose_trajectory"] = df
                    if len(df) > 0:
                        st.success("Processing complete!")
                    else:
                        print("Couldn't process video")
            else:
                my_df = pd.read_csv(csv_reference)
                my_df['color'] = my_df['keypoint_name'].replace(color_palette)
                st.session_state["pose_trajectory"] = my_df

        


    # Layout page
    layout_columns = st.columns((1,1), gap='medium')

    if video_reference is not None:
        
        with layout_columns[0]:
            tab1, tab2 = st.tabs(["Frame Select", "Video Player"])

            with tab1:
                if "video_capture" not in st.session_state:
                    with NamedTemporaryFile(suffix="mp4") as temp:
                        temp.write(video_reference.getbuffer())
                        st.session_state["video_capture"] = cv2.VideoCapture(temp.name)
                        if not st.session_state["video_capture"].isOpened():
                            print("Error: Could not open video file.")

                # Video Stats
                fps = st.session_state["video_capture"].get(cv2.CAP_PROP_FPS)
                num_frames = int(st.session_state["video_capture"].get(cv2.CAP_PROP_FRAME_COUNT))
                total_time = fps * num_frames
                
                # Select Frame
                st.session_state["video_capture"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["video_time"] - 1)
                ret, frame = st.session_state["video_capture"].read()
                if ret:
                    frame_array = np.array(frame)[:,:,[2,1,0]]
                    st.image(frame_array)
                st.slider("Frame", 0, int(num_frames), key="video_time")
            
            with tab2:
                st.video(video_reference)
            
        selected_data = st.session_state["pose_trajectory"].loc[st.session_state["pose_trajectory"]["keypoint_name"].isin(selected_trackpoints)]

        with layout_columns[1]:
            st.markdown('#### Metrics')
            with st.container(height=500):
                text = get_metrics(selected_data, selected_plot_type)
                st.markdown(text)

        # Initialize figure in session_state only once
        # if 'scatter_plot' not in st.session_state:
        #     print("making plot")

        # Create Bokeh figure with square aspect ratio and limits
        p = figure(
            title="Joint Positions",
            # tools="pan,wheel_zoom,box_zoom,box_select,reset",
            tools="",
            # x_range=(0, 1.3),
            # y_range=(0, 1.3),
            # match_aspect=True,
            aspect_scale=1,
            match_aspect=True,
            x_axis_label='x_position', 
            y_axis_label='y_position',
            # sizing_mode="stretch_width",
            # sizing_mode='stretch_both',
            sizing_mode='stretch_width',
            # width=500,
            height=500
        )

        # p.scatter(x, y, size=8, color="magenta", alpha=0.5)
        # st.session_state["pose_trajectory"]['color'] = st.session_state["pose_trajectory"]['keypoint_name'].map({"NOSE": Category10[3][0], "RIGHT_SHOULDER": Category10[3][1], "RIGHT_WRIST": Category10[3][2]})
        
        col_source = ColumnDataSource(selected_data)

        
        # Setup javascript callback
        col_source.selected.js_on_change(
            "indices",
            CustomJS(code="""
                const indices = cb_obj.indices;
                const event = new CustomEvent("INDEX_SELECT", {detail: {indices: indices}});
                document.dispatchEvent(event);
            """)
        )
        
        p.scatter(x="x", y="y", color='color', size=10, source=col_source, alpha=0.6, legend_group='keypoint_name')
        
        xmin = ymin = -1
        xmax = ymax = 1
        if len(selected_data) > 0:
            xmin = np.min(selected_data['x'])
            xmax = np.max(selected_data['x'])
            ymin = np.min(selected_data['y'])
            ymax = np.max(selected_data['y'])

        
        #p.line(x=[0, 0], y=[ymin, ymax], line_width=3, color='black')
        #p.line(x=[xmin, xmax], y=[0, 0], line_width=3, color='black')

        # LassoSelectTool to select points
        lasso = LassoSelectTool()
        p.add_tools(lasso)

        # Hover to tell sample info (e.g. frame #)
        hover = HoverTool(tooltips=[
            ("Joint", "@keypoint_name"),
            ("Frame", "@frame"),
            ("x", "@x"),
            ("y", "@y")
        ])
        p.add_tools(hover)

        p.legend.title = "Joint"
                
        st.session_state["scatter_plot"] = p

        if 'selected_indices' not in st.session_state:
            st.session_state["selected_indices"] = []
        

        # Show in Streamlit w/ events
        result = streamlit_bokeh3_events(
            bokeh_plot=p,
            events="INDEX_SELECT",
            key="foo",
            refresh_on_update=True,
            debounce_time=0
        )

        if result and "INDEX_SELECT" in result:
            st.session_state["selected_indices"] = result["INDEX_SELECT"]["indices"]
            print("Selected indices:", st.session_state["selected_indices"])
        # Render
        #streamlit_bokeh(p)
        

        # setup timeline plot
        ps = figure(
            title="Coronal Speed for Each Joint", 
            x_axis_label='Frame', 
            y_axis_label='Speed', 
            tools="pan,box_zoom,reset",
            height=150,

        )
        
        # Get subset of trajectory from positional selection
        if len(st.session_state["selected_indices"]) == 0:
            trajectories_subset = selected_data
        else:
            trajectories_subset = selected_data.loc[st.session_state["selected_indices"]]

        
        # Draw all joint lines
        for joint in trajectories_subset['keypoint_name'].unique():
            joint_data = trajectories_subset[trajectories_subset['keypoint_name'] == joint]

            # Find splits where frame is not consecutive
            joint_data = joint_data.reset_index(drop=True)
            joint_data['group'] = (joint_data['frame'].diff() != 1).cumsum()

            # Draw a line for each consecutive segment
            for _, segment in joint_data.groupby('group'):
                color = segment['color'].iloc[0]  # Use the first color for the joint
                ps.line(x=segment['frame'], y=segment['speed_xy'], legend_label=joint, line_width=2, color=color)

        if len(selected_data) > 0:
            # Legend information
            ps.legend.title = 'Joint'
            ps.legend.location = 'top_left'

        # Render
        streamlit_bokeh(ps)

        

if __name__ == "__main__":
    run_app()
    