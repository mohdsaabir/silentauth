## Overview

This branch corresponds to the **Gesture Recognition** modality for SilentAuth.
It is implemented using **OpenCV**, **MediaPipe**, and **Scikit-learn (SVM)**.

The system extracts **hand landmark features**, trains an SVM model, and performs
real-time gesture-based user authentication.

---

## Structure

### enroll_gesture.py
Registers a user by recieving gesture name and username.
This file is executed once per user.

---

### realtime_gesture.py
Verifies a user by capturing live hand landmarks and matching them
against the trained SVM gesture model.
This is the main authentication module.

---

### collected_data/
Stores extracted hand landmark feature data collected using OpenCV and Mediapipe.

---

### models/
Contains the trained SVM gesture recognition model.

---

### database/
To create the SQLite database, the script 'gesture_users.db' located in this database folder must be executed.

---

### requirements.txt
Lists all required Python dependencies for the gesture recognition module.

---

### Install Dependencies
```bash
pip install -r requirements.txt
```
---
### Register a User
```bash
python enroll_gesture.py
```
---
### Verify a User
```bash
python realtime_gesture.py
```
---
### Notes

Webcam access is required.

Enrollment must be completed before verification.

Designed to work as part of the SilentAuth system.
