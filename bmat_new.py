import streamlit as st
import altair as alt

import pandas as pd

from bokeh.io import curdoc
from streamlit_bokeh import streamlit_bokeh
from streamlit_bokeh3_events import streamlit_bokeh3_events
from bokeh.palettes import Category10
from bokeh.layouts import gridplot, column

from video_processing import *
from data_analysis import *
from graphing import *

trackpoint_choices=["NOSE", "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
color_palette = dict(zip(trackpoint_choices, Category10[len(trackpoint_choices)]))

col_source = img_source = None

def on_video_change():
    if "video_capture" in st.session_state:
        print("releasing old cap")
        st.session_state["video_capture"].release()
        del st.session_state["video_capture"]
        del st.session_state["video_time"]
        del st.session_state['bokeh_video_frames']
        del st.session_state['opencv_video_frames']

def on_csv_change():
    if 'pose_trajectory' in st.session_state:
        print("deleting old poses")
        del st.session_state["pose_trajectory"]

def run():
    global col_source
    global img_source
    print("RUN")

    st.set_page_config(
        page_title="BMAT",
        layout="wide",
        initial_sidebar_state="expanded")
    
    alt.themes.enable("dark")

    st.title('Bilateral Motion Analysis Toolkit')

    # Set up sidebar
    with st.sidebar:
        st.title('BMAT Settings')

        uploaded_vid = st.file_uploader("Choose a video... [required]", on_change=on_video_change, type=['.mp4', '.MP4', '.mov', '.MOV'])

        # New video uploaded
        if 'video_capture' not in st.session_state and uploaded_vid is not None:
            vid = load_video(uploaded_vid)
            if vid is not None:
                st.session_state["video_capture"] = vid
                st.session_state["video_time"] = 1

                st.session_state["bokeh_video_frames"], st.session_state["opencv_video_frames"] = extract_frames(vid)
                img_source = ColumnDataSource(data=dict(image=[st.session_state["bokeh_video_frames"][st.session_state["video_time"]]]))
            else:
                print("Didn't load video properly")

            
        uploaded_csv = st.file_uploader("Choose a pre-processed file... [optional]", on_change=on_csv_change, type=['.csv'])

        selected_trackpoints = st.multiselect('Select joints', trackpoint_choices)
        st.session_state['selected_trackpoints'] = selected_trackpoints

        if "pose_trajectory" not in st.session_state:
        # initialize empty 
            st.session_state["pose_trajectory"] = pd.DataFrame(columns=['frame', 'keypoint_name', 'x', 'y', 'z', 'color', 'prev_x', 'prev_y', 'speed_xy'])
        


        # Already processed
        if len(st.session_state["pose_trajectory"]) > 0 and uploaded_vid is not None:
            i = uploaded_vid.name.rfind('.')
            filename = 'BMAT_'+uploaded_vid.name[:i]+'.csv'
            st.download_button('Download pose data', st.session_state["pose_trajectory"].to_csv().encode("utf-8"), mime="text/csv", file_name=filename)
        
        # Need to process
        elif uploaded_vid is not None and uploaded_csv is None:
            if st.button('Extract pose'):
                print("processing...")
                progress_bar = st.progress(0)
                df = process_video(progress_bar, st.session_state)
                st.session_state["pose_trajectory"] = df
                if len(df) > 0:
                    st.success("Processing complete!")
                else:
                    print("Couldn't process video")
                
                st.rerun()
        
        # Pre-processed
        elif uploaded_csv is not None:
            my_df = pd.read_csv(uploaded_csv)
            st.session_state["pose_trajectory"] = my_df

    st.session_state['pose_trajectory']['color'] = st.session_state['pose_trajectory']['keypoint_name'].replace(color_palette)
    sub_df = st.session_state["pose_trajectory"].loc[st.session_state["pose_trajectory"]['keypoint_name'].isin(selected_trackpoints)]
    
    # Create shared data source for the graphs
    col_source = ColumnDataSource(sub_df)
    #TODO: see if I can get this hooked up
    # if 'bokeh_video_frames' in st.session_state:
    #     col_source.selected.js_on_change('indices', CustomJS(args=dict(imgs=img_source, viddata=st.session_state['bokeh_video_frames'], data=col_source), code="""
    #     const inds = data.selected.indices
    #     if (inds.length > 0) {
    #         const first = min(data.data[inds].frame)
    #         imgs.data = viddata[first]
    #     }
    #     """))

    # Layout page
    layout_columns = st.columns((1,1), gap='medium')
    with layout_columns[0]:
        tab1, tab2 = st.tabs(["Select Frame", "Video Player"])

        with tab1:
            if "video_time" not in st.session_state:
                st.session_state["video_time"] = 1

            if 'video_capture' in st.session_state and st.session_state['video_capture'] is not None:
                readcap = st.session_state["video_capture"]
                
                if 'video_time' in st.session_state and 'bokeh_video_frames' in st.session_state and st.session_state["video_time"] < st.session_state["bokeh_video_frames"].shape[0]:
                    img = st.session_state["bokeh_video_frames"][st.session_state["video_time"]]
                    img_p = figure(tools='', aspect_ratio=img.shape[1]/img.shape[0], background_fill_alpha = 0.0, background_hatch_alpha = 0.0)
                    img_p.yaxis.major_label_text_alpha = 0.0
                    img_p.xaxis.major_label_text_alpha = 0.0
                    img_p.x_range.range_padding = img_p.y_range.range_padding = 0

                    img_p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10) #TODO: link img_source here so it can be hooked to the slider

                    streamlit_bokeh(img_p)
                else:
                    print("Couldn't find frame at ", st.session_state["video_time"])
                
                num_frames = int(readcap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.slider("Frame", 0, int(num_frames), key="video_time")
        with tab2:
                st.video(uploaded_vid)
    
    with layout_columns[1]:
        st.markdown('#### Metrics')
        with st.container(height=500):
            text = get_metrics(sub_df, None)
            st.markdown(text)


    plot_tools = "lasso_select,reset,pan,zoom_in,zoom_out"

    scatter_p = create_scatter(col_source, plot_tools)
    speed_p = create_speed_time(col_source, plot_tools)
    
    total_p = column([scatter_p, speed_p])
    
    # Setup javascript callback
    # col_source.selected.js_on_change(
    #     "indices",
    #     CustomJS(
    #         args=dict(source=col_source),
    #         code="""
            
    #         """,
    #     ),
    # )
    # print(col_source.selected)
    # result = streamlit_bokeh3_events(
    #         bokeh_plot=total_p,
    #         events="INDEX_SELECT",
    #         key="foo",
    #         refresh_on_update=False,
    #         debounce_time=100
    #     )
    # if result is not None and "INDEX_SELECT" in result:
    #     # Need the index in the full dataset not just the index in the graphed data
    #     print("Heard a callback")
    #     frames = col_source.data[result["INDEX_SELECT"].get("indices", [])]['frame']
    #     #st.session_state["selected_frames"] = [int(x) for x in (sub_df.index[result["INDEX_SELECT"]["indices"]])]
    #     print("Selected frames:", frames)
    #     # TODO: set video time to first of these frames
    
    streamlit_bokeh(total_p)

if __name__ == "__main__":
    run()