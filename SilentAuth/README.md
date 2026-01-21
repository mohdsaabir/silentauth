##SilentAuth 🔐

Multi-Modal Continuous Authentication System

##Overview

**SilentAuth** is a multi-modal biometric authentication system that verifies users
using Face, Voice, and Gesture recognition.

Each modality is implemented as an independent module, and a central
orchestration layer coordinates enrollment and verification in a sequential manner.

The system is designed for continuous authentication and academic research,
with modular components that can be extended or replaced independently.

##System Modalities

**Face Recognition** – Visual biometric authentication

**Voice Recognition** – Speaker-based authentication

**Gesture Recognition** – Hand gesture-based authentication

All modules communicate via APIs and are triggered sequentially
during enrollment and verification.

##Installation
Install Dependencies
```bash
pip install -r requirements.txt
```
##Module Setup

Each biometric module must be set up and run independently.

Face Recognition Module
Description

Implements face-based user authentication using computer vision techniques.
The module captures facial features during enrollment and verifies users
during authentication.

Setup
```bash
cd face_module
pip install -r requirements.txt

python face_api.py

uvicorn face_api:app --port 5001
```
##Voice Recognition Module
Description

Performs speaker verification by extracting voice embeddings and comparing
them with enrolled voice samples.

Setup
```bash
cd voice_module
pip install -r requirements.txt

python voice_api.py

uvicorn voice_api:app --port 5002
```
##Gesture Recognition Module
Description

This module performs gesture-based authentication using
OpenCV, MediaPipe, and Scikit-learn (SVM).

Hand landmark features are extracted in real time, an SVM model is trained
during enrollment, and authentication is performed using live gesture input.

Registers a user by receiving gesture name and username.
Executed once per user.

Verifies a user by capturing live hand landmarks and matching them
against the trained SVM model.
setup
```bash
cd gesture_module

pip install -r requirements.txt

python gesture_api.py

uvicorn gsture_api:app --port 5003
```



Before enrollment, initialize the database:
```bash
python db_setup.py
```

##Enrollment Process (Sequential)

Enroll a user across all modalities using the orchestrator.
```bash
python orchestra_enroll.py
```

This performs:

Face enrollment

Voice enrollment

Gesture enrollment

##Verification Process (Sequential)

**Face**
```bash
python face_api_verify.py

uvicorn face_api_verify:app --port 5001
```
**Voice**
```bash
python voice_api_verify.py

uvicorn voice_api_verify:app --port 5002
```
**Gesture**
```bash
python gesture_api_verify.py

uvicorn gesture_api_verify:app --port 5003
```
```bash
python orchestra_verify.py
```

This performs:

Face verification

Voice verification

Gesture verification

##Notes

Webcam access is required for Face and Gesture modules

Microphone access is required for Voice module

Enrollment must be completed before verification

Each module must be running before orchestration

Designed to work as part of the SilentAuth system

##Technologies Used

Python

FastAPI

OpenCV

MediaPipe

Scikit-learn

REST APIs

SQLite