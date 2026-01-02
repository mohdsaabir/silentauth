Gesture Recognition Module



This module implements gesture-based authentication for the SilentAuth system.

It uses OpenCV and MediaPipe to extract hand landmarks and an SVM classifier to perform gesture recognition in real time.





📁 Project Structure



GestureRecognition/

├── collected\_data/ # Hand landmark feature data

├── database/

│ └── gesture\_users.db # SQLite database for registered users

├── models/ # Trained SVM gesture model

├── enroll\_gesture.py # User gesture enrollment script

├── realtime\_gesture.py # Real-time gesture verification

├── requirements.txt # Python dependencies

└── README.md





\### `enroll\_gesture.py`

\- Captures hand landmarks using OpenCV + MediaPipe

\- Stores landmark features for the user

\- Updates the SQLite database

\- Used for \*\*initial user registration\*\*



---



\### `realtime\_gesture.py`

\- Captures live hand landmarks from webcam

\- Loads the trained SVM model

\- Verifies the user’s gesture in real time

\- Outputs authentication result



---



\### `models/`

\- Contains the trained \*\*SVM gesture recognition model\*\*

\- Loaded during real-time verification



---



\### `collected\_data/`

\- Stores extracted \*\*hand landmark feature vectors\*\*

\- Used for training and updating the SVM model



---



\### `database/gesture\_users.db`

\- SQLite database

\- Stores user IDs and gesture metadata

\- Automatically updated during enrollment



---



\### `requirements.txt`

Contains required Python libraries:

opencv-python

mediapipe

numpy

scikit-learn



yaml

Copy code



---



\## ⚙️ Setup \& Installation



\### 1. Create virtual environment

```bash

python -m venv venv

2\. Activate virtual environment (Windows)

bash

Copy code

venv\\Scripts\\activate

3\. Install dependencies

bash

Copy code

pip install -r requirements.txt

▶️ How to Run

Step 1: Enroll a user gesture

bash

Copy code

python enroll\_gesture.py

Run once per user.



Step 2: Verify gesture in real time

bash

Copy code

python realtime\_gesture.py

Requires prior enrollment.



🔄 Execution Flow

sql

Copy code

Landmark Capture → SVM Training → User Enrollment → Real-Time Verification

📌 Notes

Webcam access is required



Enrollment must be completed before verification









