
from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool, CustomJS

def create_scatter(col_source, tools):

    p = figure(
        title="Joint Positions",
        tools=tools,
        aspect_scale=1,
        match_aspect=True,
        x_axis_label='x_position', 
        y_axis_label='y_position',
        sizing_mode='stretch_width',
        height=500
    )


    p.scatter(x="x", y="y", color='color', size=10, source=col_source, alpha=0.6, legend_group='keypoint_name')
    hover = HoverTool(tooltips=[
            ("Joint", "@keypoint_name"),
            ("Frame", "@frame"),
            ("x", "@x"),
            ("y", "@y")
        ])
    p.add_tools(hover)
    
    return p

def create_speed_time(col_source, tools):
    # setup timeline plot
    p = figure(
        title="Coronal Speed for Each Joint", 
        x_axis_label='Frame', 
        y_axis_label='Speed', 
        tools=tools,
        height=250,
        sizing_mode='stretch_width',
    )
    hover = HoverTool(tooltips=[
            ("Joint", "@keypoint_name"),
            ("Frame", "@frame"),
            ("speed", "@speed_xy")
        ])
    p.add_tools(hover)
    p.scatter('frame', 'speed_xy', source=col_source, legend_group='keypoint_name', size=6, color = 'color', alpha=0.6)
    
    return p