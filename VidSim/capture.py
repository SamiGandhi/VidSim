import util
import cv2
from parameters import Parameters as para



def capture_frames():
    # Open the video file for reading
    video_capture = cv2.VideoCapture(para.captured_video_path)
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file")
        exit()
    video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    capture_steps = int(video_fps/para.fps)
    util.delete_folder_content(para.captured_frame_dir)
    frame_nb = 0
    frame_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1, frame_number,capture_steps):
        frame_nb += 1
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        # Read the frame at the specified position
        ret, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape
        if height != para.default_height or width != para.default_width:
           frame = cv2.resize(frame, (para.default_height, para.default_width))
        image_name = para.captured_frame_dir + "/frame" + str(frame_nb) + ".png"
        cv2.imwrite(image_name,frame)
        
    video_capture.release()
    return frame_nb


def get_captured_frame(frame_nb):
    image_name = para.captured_frame_dir + "/frame" + str(frame_nb) + ".png"
    frame = cv2.imread(image_name)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame



def capture_frames_original_fps():
    # Open the video file for reading
    video_capture = cv2.VideoCapture(para.captured_video_path)
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file")
        exit()
    video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    util.delete_folder_content(para.captured_frame_dir)
    frame_nb = 0
    frame_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = video_capture.read() 
        if not ret:
            break
        frame_nb += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape
        if height != para.default_height or width != para.default_width:
           frame = cv2.resize(frame, (para.default_height, para.default_width))
        image_name = para.captured_frame_dir + "/frame" + str(frame_nb) + ".png"
        cv2.imwrite(image_name,frame)
    video_capture.release()
    return frame_nb




class Captured_Frame:
    def __init__(self,frame,frame_nb,frame_type):
        self.frame = frame
        self.frame_number = frame_nb
        self.frame_type = frame_type
        self.reference_frame = frame_nb


