import streamlit as st
from streamlit_drawable_canvas import st_canvas

import json

canvas_result = st_canvas(
    fill_color="#EA1010",
    stroke_width=3,
    height=320,
    width=512,
    drawing_mode="rect",
    key="color_annotation_app",
)
if canvas_result.json_data is not None:

    # Serializing json
    json_object = json.dumps(canvas_result.json_data, indent=4)

    # Writing to sample.json
    with open("canvas.json", "w") as outfile:
        outfile.write(json_object)