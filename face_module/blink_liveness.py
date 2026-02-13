import numpy as np

class BlinkDetector:
    def __init__(self, threshold=2.5):
        self.prev_dist = None
        self.blinked = False
        self.threshold = threshold

    def reset(self):
        self.prev_dist = None
        self.blinked = False

    def update(self, kps):
        # Use eye + nose vertical relation (works with 5 points)
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]

        eye_mid_y = (left_eye[1] + right_eye[1]) / 2
        dist = abs(eye_mid_y - nose[1])

        if self.prev_dist is not None:
            if abs(dist - self.prev_dist) > self.threshold:
                self.blinked = True

        self.prev_dist = dist
        return self.blinked
