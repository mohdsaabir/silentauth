## SilentAuth 🔐

Multi-Modal Biometric Authentication System

## Overview

**SilentAuth** is a multi-modal biometric authentication system that verifies users
using Face, Voice, and Gesture recognition.

Each modality is implemented as an independent module, and a central
Execution Engine layer coordinates enrollment and verification.The Enrollment is 
sequential and verification is by means of parallel processing.

The system is designed for continuous authentication and academic research,
with modular components that can be extended or replaced independently.

## System Modalities

**Face Recognition** – Visual biometric authentication

**Voice Recognition** – Speaker-based authentication

**Gesture Recognition** – Hand gesture-based authentication

All modules communicate via APIs and are triggered sequentially
during enrollment and verification.

**Preset Management** – Stores and manages pre-configured user authentication profiles
for each user. Presets contain user-specific thresholds, model references, and 
weights for Face, Voice, and Gesture modules. 

Presets are automatically created during enrollment and applied during verification 
to optimize recognition accuracy. Users can update, delete, or select presets via 
the preset manager.

## Installation & Setup

Strictly follow these instructions to have a hussle-free setup of the system

Step 1 : In the root folder , create a virtual environment 'venv' (any python version)

Step 2 : In the face_module folder, create a virtual environment 'venv' (python 3.11.9)

Step 3 : In the gesture_module folder, create a virtual environment 'venv' (python 3.10.0)

step 4 : In the voice_module folder, create a virtual environment 'venv' (python 3.10.0)

step 5 : Install Dependencies on each of these four virtual environments 'venv'
```bash
pip install -r requirements.txt
```
step 6 : Setup the database.(In the root of the repo)
```bash
python db_setup.py
```
Setup complete.

## Enrollment Process

## Face Recognition Module
Description

Implements face-based user authentication using computer vision techniques.
The module captures facial features during enrollment and verifies users
during authentication.

## Voice Recognition Module
Description

Performs speaker verification by extracting voice embeddings and comparing
them with enrolled voice samples.

## Gesture Recognition Module
Description

This module performs gesture-based authentication using
OpenCV, MediaPipe, and Scikit-learn (SVM).

Hand landmark features are extracted in real time, an SVM model is trained
during enrollment, and authentication is performed using live gesture input.

Registers a user by receiving gesture name and username.
Executed once per user.

Verifies a user by capturing live hand landmarks and matching them
against the trained SVM model.

- Preset creation and storage for the user


## How to Execute Multimodal Enrollment?
(make sure all terminals are closed)

Step 1 : ctrl + shift + p

Step 2 : In the window , type 'Tasks : Run Task' then click it.

Step 3 : Click 'SilentAuth-Enrollment (ALL)' and then click 'continue without scanning'

step 4 : In the four terminals opened , wait till all services are active , then in the active terminal run : 
```bash
python multimodal_enroll.py
```
step 5 : enrolment begins.

This performs:

Face enrollment

Voice enrollment

Gesture enrollment

Preset creation and storage for the user

## Verification Process 

## Camera Service (Producer)

A single camera service captures live video frames using OpenCV.

Frames are published via ZeroMQ (PUB) so multiple modules can use the same stream.

This avoids opening the camera multiple times and keeps latency low.

Face API and Gesture API act as ZMQ SUBscribers.

They receive only the latest frame (using CONFLATE) for real-time processing.

Ensures smooth parallel execution without frame backlog.

## Face Authentication API

Extracts facial features from incoming frames.

Performs 1-vs-N face recognition to identify the user.

Outputs a username + confidence score.

## Gesture Authentication API

Detects hand/body gestures from the same video stream.

Classifies gestures using a trained model (MediaPipe + ML).

Outputs a gesture match score.

## Voice Authentication API

Runs independently using microphone input.

Verifies the speaker based on voice features.

Outputs a voice confidence score.

## Fusion Engine (Final Decision)

Collects results from Face, Gesture, and Voice APIs.

Applies weighted decision fusion, using preset-specific weights if available.

If the combined score crosses the threshold, access is granted.

User-specific presets ensure consistent weighting, faster verification, and 
optimized accuracy for enrolled users.




## How to Execute Multimodal Verification?
(make sure all terminals are closed)

Step 1 : ctrl + shift + p

Step 2 : In the window , type 'Tasks : Run Task' then click it.

Step 3 : Click 'SilentAuth (ALL)' and then click 'continue without scanning'

step 4 : In the five terminals opened , wait till all services are active , then in the active terminal run : 
```bash
python fusion_engine.py
```
step 5 : verification begins.

This performs:

Face verification

Voice verification

Gesture verification

Fusion 

Authentication/Rejection

## Notes

Webcam access is required for Face and Gesture modules

Microphone access is required for Voice module

Enrollment must be completed before verification

Each module must be running before running executor files in active terminals

Designed to work as part of the SilentAuth system


## System Testing & Evaluation 🔬

SilentAuth also includes internal tools used for system testing, statistical evaluation, and parameter optimization.
These scripts were used to generate experimental data and determine the final configuration of the authentication system.

These files are not required for normal enrollment or verification but are used during the research and evaluation phase of the system.

## Analysis Scripts (AN_*)

Files prefixed with AN_ are analysis utilities used to evaluate the performance of the multimodal authentication pipeline.

These scripts help determine:

Optimal fusion weights

Optimal fusion thresholds

Voice contradiction penalty tuning

Attack simulation and log generation

Dataset preparation for evaluation

Included Analysis Files

### AN_generator.py

Generates synthetic multimodal attack scenarios and evaluation samples used to simulate cross-user and unknown-user authentication attempts.

### AN_voice_contradiction.py

Analyzes cases where the voice modality contradicts the face identity and determines the optimal contradiction penalty threshold to prevent impersonation attacks.

### AN_fusion_weight.py

Performs parameter sweeps across different combinations of:

Face weight
Voice weight
Gesture weight

to determine the optimal weighting configuration for multimodal fusion.

### AN_fusion_threshold.py

Computes the optimal final decision threshold for the fusion engine by analyzing score distributions and evaluation metrics.

## Evaluation Datasets

The following CSV files contain experiment logs used during evaluation:

### AN_multimodal_logs_new.csv
Processed dataset used for multimodal fusion analysis.

### AN_old_logs.csv
Previous experimental logs retained for reference.

### AN_fusion_final.csv
Final dataset used for determining optimal system parameters.

## System Testing Utility

tester1.py

This script is used to perform large-scale system testing.

It runs multimodal authentication attempts and logs the following information:

Face confidence score

Voice confidence score

Gesture confidence score

Voice contradiction penalty

Final fusion score

Authentication decision

The generated logs are used to evaluate:

Genuine user performance

Cross-user impersonation attempts

Unknown user rejection

System robustness under attack scenarios

Results Folder
results/

The results folder stores outputs generated during evaluation experiments.

This may include:

ROC curve plots

Score distribution histograms

Evaluation statistics

Threshold analysis results

Experiment logs

These artifacts are used to validate the accuracy, robustness, and security of the SilentAuth multimodal authentication system.