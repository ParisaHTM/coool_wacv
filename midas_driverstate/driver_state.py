import cv2
import os
import numpy as np

class Driver_State():
    def __init__(self, video_root, video_name):
        self.video_root = video_root
        self.video_name = video_name
        self.threshold = 4
        self.loop_frame = 5
        self.st = 1
        self.driver_state_flag = False
        self.p0 = None
        self.old_gray = None
        self.slope_history = []
        
        # Parameters for lucas kanade optical flow 
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                             10, 0.03)) 

        # params for corner detection 
        self.feature_params = dict(maxCorners=100, 
                              qualityLevel=0.3, 
                              minDistance=7, 
                              blockSize=7) 

    def get_good_feature_from_first_frame(self,):
        video_stream = cv2.VideoCapture(os.path.join(self.video_root, self.video_name))
        assert video_stream.isOpened()
        
        while video_stream.isOpened():
            ret, old_frame = video_stream.read() 
            self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            if np.all(self.old_gray == 0):
                continue
            else:
                break
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params) 

    def optical_flow(self, frame_image):
        frame_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY) 
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)            
        # Select good points 
        good_new = p1[st == self.st] 
        good_old = self.p0[st == self.st] 
        
        # Calculate motion vectors and slopes
        motion_vectors = []
        # Calculate slopes for the current frame
        slopes = []
        for i, (new, old) in enumerate(zip(good_new, good_old)): 
            a, b = new.ravel() 
            c, d = old.ravel() 
            
            slope = (b - d) / ((a - c) + 1e-100)
            slopes.append(slope)
        return slopes

    def calc_slop(self, frame, frame_image):
        avg_slope_change = 0
        slopes = self.optical_flow(frame_image)
        # Update slope history every 5 frames                
        if frame % self.loop_frame == 0:
            if self.slope_history:
                previous_slopes = self.slope_history[-1]
                slope_changes = [abs(s - ps) for s, ps in zip(slopes, previous_slopes)]
                avg_slope_change = np.mean(slope_changes) 
                if avg_slope_change < self.threshold and self.driver_state_flag != True:
                    self.driver_state_flag = True
                    print(f"Frame {frame} labeled as True")           
            self.slope_history.append(slopes)  # Update the history
        return self.driver_state_flag
        