import cv2
import numpy as np

class Person:
    def __init__(self, bbox, track_id):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.track_id = track_id
        self.authorized = False
        self.marker_id = None


