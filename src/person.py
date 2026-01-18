import numpy as np

class Person:
    def __init__(self, bbox, track_id):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.track_id = track_id
        self.authorized = False
        self.marker_id = None
        self.confidence = 0.0
        self.center = self._calculate_center()
        self.age = 0  # How long this person has been tracked
        self.last_seen_frame = 0
   