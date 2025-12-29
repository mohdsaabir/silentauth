# webrtcvad/__init__.py
# Dummy stub to satisfy Resemblyzer dependency
# This will treat all audio as speech (no trimming).

class Vad:
    def __init__(self, mode=3):
        # mode 0-3 in real VAD, ignored here
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def is_speech(self, frame, sample_rate):
        # Always say this frame contains speech
        return True
