from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration


class GazeEstimation(object):

    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):

        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.coords_arr = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):

        """Check that the pupils have been located"""

        try:

            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)

            return True

        except Exception:

            return False

    def _analyze(self):

        """Detects the face and initialize Eye objects"""
        
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:

            landmarks = self._predictor(frame, faces[0])

            left_x1 = landmarks.part(36).x
            left_x2 = landmarks.part(39).x
            left_y1 = landmarks.part(37).y
            left_y2 = landmarks.part(40).y

            right_x1 = landmarks.part(42).x
            right_x2 = landmarks.part(45).x
            right_y1 = landmarks.part(44).y
            right_y2 = landmarks.part(47).y

            left_eye_bbox_coords = [left_x1, left_x2, left_y1, left_y2]
            right_eye_bbox_coordds = [right_x1, right_x2, right_y1, right_y2]

            self.coords_arr = (left_eye_bbox_coords, right_eye_bbox_coordds)

            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:

            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):

        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        
        self.frame = cv2.flip(frame, 1)
        self._analyze()

    def pupil_left_coords(self):

        """Returns the coordinates of the left pupil"""
        
        if self.pupils_located:

            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            
            return (x, y)

    def pupil_right_coords(self):

        """Returns the coordinates of the right pupil"""
        
        if self.pupils_located:

            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            
            return (x, y)

    def horizontal_ratio(self):

        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        
        if self.pupils_located:

            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):

        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        
        if self.pupils_located:

            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            
            return (pupil_left + pupil_right) / 2

    def is_right(self):

        """Returns true if the user is looking to the right"""
        
        if self.pupils_located:
            
            return self.horizontal_ratio() <= 0.35

    def is_left(self):

        """Returns true if the user is looking to the left"""
        
        if self.pupils_located:

            return self.horizontal_ratio() >= 0.65

    def is_center(self):

        """Returns true if the user is looking to the center"""
        
        if self.pupils_located:

            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):

        """Returns true if the user closes his eyes"""

        if self.pupils_located:

            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            
            return blinking_ratio > 3.8

    def annotated_frame(self):

        """Returns the main frame with pupils highlighted"""
        
        frame = self.frame.copy()

        if self.pupils_located:

            color = (255, 255, 255)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()

            acc_x, acc_y = int((x_left + x_right)/2), int((y_left + y_right)/2)

            cv2.line(frame, (x_left - 1, y_left), (x_left + 1, y_left), color)
            cv2.line(frame, (x_left, y_left - 1), (x_left, y_left + 1), color)
            
            cv2.line(frame, (x_right - 1, y_right), (x_right + 1, y_right), color)
            cv2.line(frame, (x_right, y_right - 1), (x_right, y_right + 1), color)

        return frame

    def eye_roi_bb(self):

        return self.coords_arr

    def get_frame_size(self):

        width = self.frame.shape[1]
        height = self.frame.shape[0]

        return width, height