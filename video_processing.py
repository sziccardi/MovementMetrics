
import cv2
from tempfile import NamedTemporaryFile

from io import StringIO
from csv import writer
import pandas as pd
import numpy as np
from tqdm import tqdm

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def load_video(vid_buffer):
    with NamedTemporaryFile(suffix="mp4") as temp:
            temp.write(vid_buffer.getbuffer())
            vid = cv2.VideoCapture(temp.name)
            if not vid.isOpened():
                print("Error: Could not open video file.")
                return None
            return vid

def cv2_to_bokeh(frame):
    M, N, _ = frame.shape
    img = np.empty((M, N), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((M, N, 4))
    view[:,:,0] = frame[:,:,2] # copy red channel
    view[:,:,1] = frame[:,:,1] # copy blue channel
    view[:,:,2] = frame[:,:,0] # copy green channel
    view[:,:,3] = 255

    img = img[::-1] # flip for Bokeh

    return img, M, N


def extract_frames(video):
    bokeh_frames = []
    opencv_frames = []
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(num_frames)):
        if video.isOpened():
            # Read frame from video
            success, image = video.read()
            if not success:
                print("Ignoring empty camera frame at frame #", frame_num)
                continue

            img, M, N = cv2_to_bokeh(image)
            bokeh_frames.append(img)
            opencv_frames.append(image)
    bokeh_frames = np.array(bokeh_frames)
    print(bokeh_frames.shape)
    opencv_frames = np.array(opencv_frames)
    print(opencv_frames.shape)
    return bokeh_frames, opencv_frames
     

def process_video(progress_bar, session_state):
    output = StringIO()
    csv_writer = writer(output)
    csv_writer.writerow(['frame', 'keypoint_name', 'x', 'y', 'z', 'color', 'prev_x', 'prev_y', 'speed_xy'])

    if "opencv_video_frames" in session_state:
        #readcap = session_state["video_capture"]
        with mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                
                num_frames = len(session_state['opencv_video_frames'])

                for frame_num in tqdm(range(num_frames)):
                    # Process frame with mediapipe
                    image = session_state['opencv_video_frames'][frame_num]
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    # Write the landmark positions to df
                    if results.pose_landmarks is not None:
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            new_row = [frame_num, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z]
                            
                            csv_writer.writerow(new_row)

                    progress_bar.progress(frame_num / num_frames)

        output.seek(0)
        df = pd.read_csv(output)
        if len(df) == 0:
            print("DIDNT PROCESS PROPERLY")
            return df

        # Center positions around sternum and flip vertically
        LShoulder = df.loc[df['keypoint_name'] == "LEFT_SHOULDER"]
        LShoulder = LShoulder.set_axis([f for f in LShoulder['frame']])
        LShoulder = LShoulder[['x', 'y', 'z']]

        RShoulder = df.loc[df['keypoint_name'] == "RIGHT_SHOULDER"]
        RShoulder = RShoulder.set_axis([f for f in RShoulder['frame']])
        RShoulder = RShoulder[['x', 'y', 'z']]

        sternum = LShoulder.add(RShoulder) / 2.0

        keypoints = df['keypoint_name'].unique()
        for key in keypoints:
            keything = df.loc[df['keypoint_name'] == key, ['x', 'y', 'z']]
            keysternum = sternum.set_axis([f for f in keything.index])

            df.loc[df['keypoint_name'] == key,['x', 'y', 'z']] = (keything - keysternum)
            df.loc[df['keypoint_name'] == key,'y'] = df.loc[df['keypoint_name'] == key,'y'] * -1.0

        
        # Compute speed
        df['prev_x'].iloc[1:] = df['x'].iloc[:-1]
        df['prev_y'].iloc[1:] = df['y'].iloc[:-1]

        dx = df['x'].iloc[1:] - df['prev_x'].iloc[1:]
        dy = df['y'].iloc[1:] - df['prev_y'].iloc[1:]
        fps = 30
        if 'video_capture' in session_state:
            fps = int(session_state['video_capture'].get(cv2.CAP_PROP_FPS))
        df['speed_xy'].iloc[1:] = np.sqrt(dx*dx + dy*dy) * fps
        
        # Return clean dataframe
        progress_bar.progress(1.0)
        return df
    
    else: 
        output.seek(0)
        df = pd.read_csv(output)
        return df