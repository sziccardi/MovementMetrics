import numpy as np
import cv2 as cv

vid_loc = "C:/Users/Zicca/Box/AHA videos/RoboCP3_3mo.m4v"
i = vid_loc.rfind('.')
out_loc = vid_loc[:i] + ".mp4"
cap = cv.VideoCapture(vid_loc)
fps = int(cap.get(cv.CAP_PROP_FPS))
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
out = cv.VideoWriter(out_loc, 0x00000021, fps, (w, h)) 
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
 
    # write the frame
    out.write(frame)
 
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()