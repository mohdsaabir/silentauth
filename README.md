##  Project Structure

This repository implements a **speaker enrollment and verification system** using voice embeddings.
It uses resemblyzer model.
Below is a brief description of each file and its purpose.

---

### `db_utils.py`
Contains all **database-related utilities**.

---

### `enroll.py`
Used for **live speaker enrollment via microphone**.

- Records multiple voice samples from the user
- Extracts speaker embeddings using the voice encoder
- Averages and normalizes embeddings
- Saves embeddings locally in embeddings folder and in the database

---

### `enroll_from_files.py`
Used to **enroll a speaker using existing audio files**.

- Reads `.wav` files from folder data/enroll/
- Processes and embeds each audio file
- Creates a single representative embedding for the user
- Stores the result locally in embeddings folder and in the database

---

### `verify.py`
Performs **live speaker verification via microphone input**.

- Records a voice sample
- Extracts an embedding
- Compares it against stored embeddings
- Uses similarity scoring to verify identity

---

### `verify_file_batch.py`
Performs **batch verification** on multiple audio files.

- Processes the audio files in folder data/verify
- Verifies each file against stored speakers
- Outputs verification results for all files

---

### `requirements.txt`
Lists all required Python dependencies.

---

### `webrtcvd`
Bypasses the need of vs build tools, dependencies related to resemblyzer.

---

### `enrolled audio`
Contains the audio files obtained from microphone through enroll.py and verify.py

---

### `data`
Contains two folder enroll and verify, can be used to store prerecorded audio clip for testing.
In enroll the audio of user must be stored in corresponding user folder name

---

### `database`
Should create a db named voice_embeddings

---

### `Directory Structure`
<img width="280" height="435" alt="Screenshot 2025-12-21 172228" src="https://github.com/user-attachments/assets/4b4bd5d1-69b7-47c9-a8ba-2360f16231ba" />

---

Install dependencies using:
```bash
pip install -r requirements.txt


