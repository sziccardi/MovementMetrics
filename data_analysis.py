import pandas as pd

def get_metrics(df, plot_type):
    metrics_text = ""
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