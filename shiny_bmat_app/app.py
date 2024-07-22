from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from pathlib import Path
here = Path(__file__).parent

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csvfile", "Choose Pose CSV File", accept=[".csv"], multiple=False),
        ui.input_file("videofile", "and Associated Video File", accept=[".mp4"], multiple=False),
        ui.input_selectize(
        "track_points",
        "Choose track points to plot:",
        multiple=True,
        choices=({
            "Face": {"nose": "Nose", "left_eye(inner)": "Inner left eye", "left_eye": "Center left eye", "left_eye(outer)": "Outer left eye", "right_eye(inner)": "Inner right eye", "right_eye": "Center right eye", "right_eye(outer)": "Outer right eye", "right_ear": "Right ear", "left_ear": "Left ear", "mouth(right)": "Right mouth corner", "mouth(left)": "Left mouth corner"},
            "Upper Body": {"left_wrist": "Left wrist", "left_elbow":"Left elbow", "left_shoulder": "Left shoulder", "right_shoulder": "Right shoulder", "right_elbow":"Right elbow", "right_wrist":"Right wrist"},
            "Right Hand": {"right_thumb_base": "Base of thumb", "right_thumb1": "Proximal thumb joint", "right_thumb2": "Distal thumb joint", "right_thumb_tip":"Tip of thumb",
            "right_index_base": "Base of index finger", "right_index1": "Proximal index joint", "right_index2": "Distal index joint", "right_index_tip":"Tip of index finger","right_middle_base": "Base of middle finger", "right_middle1": "Proximal middle joint", "right_middle2": "Distal middle joint", "right_middle_tip":"Tip of middle finger","right_ring_base": "Base of ring finger", "right_ring1": "Proximal ring joint", "right_ring2": "Distal ring joint", "right_ring_tip":"Tip of ring finger","right_pinky_base": "Base of pinky finger", "right_pinky1": "Proximal pinky joint", "right_pinky2": "Distal pinky joint", "right_pinky_tip":"Tip of pinky finger"},
            "Left Hand": {"left_thumb_base": "Base of thumb", "left_thumb1": "Proximal thumb joint", "left_thumb2": "Distal thumb joint", "left_thumb_tip":"Tip of thumb",
            "left_index_base": "Base of index finger", "left_index1": "Proximal index joint", "left_index2": "Distal index joint", "left_index_tip":"Tip of index finger","left_middle_base": "Base of middle finger", "left_middle1": "Proximal middle joint", "left_middle2": "Distal middle joint", "left_middle_tip":"Tip of middle finger","left_ring_base": "Base of ring finger", "left_ring1": "Proximal ring joint", "left_ring2": "Distal ring joint", "left_ring_tip":"Tip of ring finger","left_pinky_base": "Base of pinky finger", "left_pinky1": "Proximal pinky joint", "left_pinky2": "Distal pinky joint", "left_pinky_tip":"Tip of pinky finger"},
        })
        ),
        ui.input_select(
        "plot_type",
        "Choose the type of plot:",
        {
            "plot_pos":"Relative position",
            "plot_ang":"Relative joint angle",
            "plot_vel":"Relative velocity"
        },
    ),
        ui.output_text("set_head"),
        ui.input_numeric("fps", "FPS", 30),
        ui.input_numeric("px2m", "Pixels per Meter (px)", -1),
        ui.input_numeric("smoothing", "Smoothing Amount (frames)", 1),
        # ui.input_action_button("plot_button", "PLOT") 
    ),
    ui.panel_title("Bilateral Motion Analysis Toolkit", "BMAT"),
    ui.layout_columns(
     ui.card(
        ui.output_text("video_name"),
        ui.output_image("display_frame"),
        ui.input_slider("frame_select", "", min=0, max=1, value=0, step=1),
        ),
     ui.card(ui.card_header("Metrics"),
             ui.output_text_verbatim("display_metrics"))
    ),
    ui.card(ui.card_header("Point cloud"),
            ui.output_plot("point_cloud")),
    ui.card(ui.card_header("Over time"),
            ui.output_plot("over_time"))
)


def server(input, output, session):

    @render.text
    def set_head():
        return "Settings:"
    
    @render.text
    def video_name():
        file = input.videofile()
        if file is None:
            return ""
        else:
            return file[0]["name"]

    @render.image
    def display_frame():
        file = input.videofile()
        if file is None:
            return {"src": here/"TEMP.png", "width": "100%"} 
        else:
            
            ui.update_slider(
                "frame_select",
                value=max(min(input.frame_select(), input.max()), input.min()),
                min=input.min(),
                max=input.max(),
            )
            {"src": here/"TEMP.png", "width": "100%"} 


    @render.text
    def display_metrics():
        df = load_data()

        total_track_point_text = ""
        fps = (float)(input.fps())

        my_points = input.track_points()
        labels = [k for k in my_points]

        plot_type = input.plot_type()

        for i, name in enumerate(labels):
            
            if plot_type == "plot_pos":
                
                # Clear less confident values
                # np_data = np.array(list(dict_data.values()))
                # conf = mm.GetPlotSpecificInfo("relative position")[0]
                # data_filter = np_data[:,-1] > conf

                # Compute crosses
                y_df = df[name+"_y_norm"]
                y_df_shifted = y_df.shift(periods=1)
                y_df_shifted = y_df_shifted[1:]
                y_df = y_df[1:]

                x_ax_crosses_count = sum((y_df_shifted < 0) & (y_df > 0)) + sum((y_df_shifted > 0) & (y_df < 0))
                x_ax_crosses_time = 0
                if name in ["nose", "left_eye(inner)", "left_eye", "left_eye(outer)", "right_eye(inner)", "right_eye", "right_eye(outer)", "right_ear", "left_ear", "mouth(right)", "mouth(left)", "left_shoulder", "right_shoulder"]:
                    x_ax_crosses_time = sum(y_df < 0) / fps
                else:
                    x_ax_crosses_time = sum(y_df > 0) / fps
            

                x_df = df[name+"_x_norm"]
                x_df_shifted = x_df.shift(periods=1)
                x_df_shifted = x_df_shifted[1:]
                x_df = x_df[1:]

                y_ax_crosses_count = sum((x_df_shifted > 0) & (x_df < 0)) + sum((x_df_shifted < 0) & (x_df > 0))
                y_ax_crosses_time = 0
                if 'left' in name:
                    y_ax_crosses_time = sum(x_df > 0) / fps
                elif 'right' in name:
                    y_ax_crosses_time = sum(x_df < 0) / fps
                
                try:
                    
                    total_track_point_text = total_track_point_text + "\n" + name + " : " +  "\n - crossed body midline " + str(int(y_ax_crosses_count/2.0)) + " times\n - " + str(round(y_ax_crosses_time,2)) + " sec spent crossed\n"
                except:
                    print("WARNING: couldn't display midline cross counts")
                
                
                try:
                    total_track_point_text = total_track_point_text + " - raised above shoulders " + str(int(x_ax_crosses_count/2.0)) + " times\n - " + str(round(x_ax_crosses_time,2)) + " sec spent raised\n"
                except:
                    print("WARNING: couldn't display shoulder cross counts")
                
                # Compute averages and spread
                data_x_mean = df[name+'_x'].mean()
                data_y_mean = df[name+'_y'].mean()
                data_x_var = df[name+'_x'].std()
                data_y_var = df[name+'_y'].std()
                try:
                    total_track_point_text = total_track_point_text + " - average position ( " + str(round(data_x_mean,2)) + ", " + str(round(data_y_mean,2)) + " )\n"
                    total_track_point_text = total_track_point_text + " - with std of ( "+ str(round(data_x_var,2)) + ", " + str(round(data_y_var,2)) + " )\n"
                except:
                    print("WARNING: couldn't display means or variances ")
                

                # included = sum(data_filter)
                # num_skipped = np_data[:,-1].shape[0] - included
                # selected = np_data[data_filter]
                # avg_conf = np.mean(selected[:,-1])
                # try:
                #     total_track_point_text = total_track_point_text + "# frames skipped: "+str(num_skipped) + "\n"
                # except:
                #     print("WARNING: couldn't display skipped frame count")
                # try:
                #     total_track_point_text = total_track_point_text + "Average confidence: "+str(round(avg_conf,2)) + "\n"
                # except:
                #     print("WARNING: couldn't display average confidence value")
            # elif "angle" in plot_type:
            #     np_data = np.array(list(data[0][i].values()))
            #     info = mm.GetPlotSpecificInfo("relative angle over time")
            #     conf_thresh = info[0]
            #     data_filter = np_data[:,-1] > conf_thresh
            #     ext_thresh = info[1]
            #     cont_thresh = info[2]
            #     temp_total_count, temp_num_count = mm.getValueCrossedCounts(np_data[data_filter,1], False, ext_thresh)
            #     try:
            #         total_track_point_text = total_track_point_text + "\n" + name + " : \n - fully extended " + str(temp_num_count) + " times\n - " + str(round(temp_total_count / fps,2)) + " sec spent fully extended\n"
            #     except:
            #         print("WARNING: couldn't display extension count")        
            #     temp_total_count, temp_num_count = mm.getValueCrossedCounts(np_data[data_filter,1], True, cont_thresh)
            #     try:
            #         total_track_point_text = total_track_point_text + " - fully tucked  " + str(temp_num_count) + " times\n - " + str(round(temp_total_count / fps,2)) + " sec spent tucked\n"
            #     except:
            #         print("WARNING: couldn't display tucked count")        
            #     data_mean = mm.getMean(np_data[data_filter,1])
            #     data_var = mm.getSTD(np_data[data_filter,1])
            #     try:
            #         total_track_point_text = total_track_point_text + " - average angle is " + str(round(data_mean,2)) + "\n"
                
            #         total_track_point_text = total_track_point_text + " - with std of "+ str(round(data_var,2)) + "\n"
            #     except:
            #         print("WARNING: couldn't display mean or variance")

        return total_track_point_text

    
    @reactive.calc
    def video_file():
        file = input.videofile()
        if file is None:
            return "TEMP.png"
        else:
            filepath = file[0]["datapath"]
            print("returning ", filepath)
            return filepath
    
    
    def load_data():
        file = input.csvfile()
        if file is None:
            return pd.DataFrame()
        
        df = pd.read_csv(file[0]["datapath"])
        mid_x = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2.0
        mid_y = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2.0

        for col in df.columns:
            if "_x" in col:
                df[col+"_norm"] = df[col] - mid_x
            elif "_y" in col:
                df[col+"_norm"] = df[col] - mid_y

        df['frame'] = df.index
        
        return df


    @render.plot
    def point_cloud():
        df = load_data()

        fig, ax = plt.subplots()
        
        if len(df) == 0:
            return fig
        
        my_points = input.track_points()
        keys = [k for k in my_points]
        
        x_keys = []
        y_keys = []
        for k in keys:
            x_keys.append(k+"_x_norm")
            y_keys.append(k+"_y_norm")
            
        plt.axhline(0, color='black', linewidth=.5)
        plt.axvline(0, color='black', linewidth=.5)
        plt.xlabel("Horizontal Position")
        plt.ylabel("Vertical Position")
        for i,k in enumerate(x_keys):
            real_label = k[:k.rfind('_x')]
            sns.scatterplot(x=x_keys[i], y=y_keys[i], data=df, ax=ax, label=real_label)
        
        if len(keys) > 0:
            ax.legend(loc="upper right", fancybox=True, ncol=1) 

        return fig

    @render.plot
    def over_time():
        df = load_data()

        fig, ax = plt.subplots(nrows=2, ncols=1)

        if len(df) == 0:
            return fig
            
        my_points = input.track_points()
        keys = [k for k in my_points]
        
        x_keys = []
        y_keys = []
        for k in keys:
            x_keys.append(k+"_x_norm")
            y_keys.append(k+"_y_norm")

        
        ax[0].axhline(0, color='black', linewidth=.5)
        ax[0].axvline(0, color='black', linewidth=.5)
        ax[0].set_xlabel("Time (frame)")
        ax[0].set_ylabel("Horizontal Position (px)")
        ax[1].axhline(0, color='black', linewidth=.5)
        ax[1].axvline(0, color='black', linewidth=.5)
        ax[1].set_xlabel("Time (frame)")
        ax[1].set_ylabel("Vertical Position (px)")
        for i,k in enumerate(x_keys):
            real_label = k[:k.rfind('_x')]
            sns.lineplot(x='frame', y=x_keys[i], data=df, ax=ax[0], label=real_label)
            sns.lineplot(x='frame', y=y_keys[i], data=df, ax=ax[1])

        if len(x_keys) > 0:
            ax[0].legend(loc="upper left", fancybox=True, ncol=1, bbox_to_anchor=(1.05, 1.0)) 
        
        return fig
    
    

    # @reactive.effect
    # @reactive.event(input.plot_button)
    # def _():
    #     # ready to plot!
    #     ui.insert_ui(
    #         ui.tags.video(id="vid"+video_id(),src=video_file(), controls=True),
    #         selector="#video_name",
    #         where="afterEnd",
    #     )
        



app = App(app_ui, server)
