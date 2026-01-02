\## Overview



This branch corresponds to the \*\*Gesture Recognition\*\* modality for SilentAuth.

It is implemented using \*\*OpenCV\*\*, \*\*MediaPipe\*\*, and \*\*Scikit-learn (SVM)\*\*.



The system extracts \*\*hand landmark features\*\*, trains an SVM model, and performs

real-time gesture-based user authentication.



---



\## Structure



\### enroll\_gesture.py

Registers a user by capturing hand landmarks using live camera input.

The extracted landmark features are stored and used for training.

This file is executed \*\*once per user\*\*.



---



\### realtime\_gesture.py

Verifies a user by capturing live hand landmarks and comparing them

against the trained SVM gesture model.

This is the \*\*main authentication module\*\*.



---



\### collected\_data/

Stores extracted \*\*hand landmark feature data\*\* collected during enrollment.

These features are used to train and update the gesture recognition model.



---



\### models/

Contains the trained \*\*SVM gesture recognition model\*\*.

The model is loaded during real-time verification.



---



\### database/

A local SQLite database (`gesture\_users.db`) is used to store user metadata.

The database is created automatically during enrollment.

Only gesture-related metadata is stored, not images or videos.



---



\### requirements.txt

Lists all required Python dependencies for the gesture recognition module.



---



\## Usage



\*\*Note:\*\* Run `enroll\_gesture.py` at least once before verification.



\### Install dependencies



```bash

pip install -r requirements.txt

Register a user

bash

Copy code

python enroll\_gesture.py

Verify a user

bash

Copy code

python realtime\_gesture.py

Notes

Webcam access is required



Enrollment must be completed before verification



This module is designed to work sequentially with other SilentAuth modalities



markdown

Copy code



