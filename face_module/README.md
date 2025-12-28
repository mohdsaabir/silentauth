## Overview

This branch corresponds to the **Face Recognition modality** for SilentAuth.
It is implemented using **InsightFace**, **OpenCV**, and **NumPy**.

---

## Structure

- **Face_Registration.py**  
  Registers a user into the local SQLite database using live camera input.  
  This file is executed **once per user**.

- **Face_Verification.py**  
  Verifies a user against the set of registered users in the database.  
  This is the **main authentication module**.

---

## Database

- A local SQLite database (`face_auth.db`) is created automatically during registration.
- The database stores **face embeddings**, not images.
- The database file is **not tracked in Git** for privacy and security reasons.

---

## requirements.txt

Lists all required Python dependencies for the face recognition module.

---

## Usage

Note: Run Face_Registration.py at least once before verification.

Install dependencies using:
```bash
pip install -r requirements.txt
```
Register a user:
```bash
python Face_Registration.py
```
Verify a user:
```bash
python Face_Verification.py
```
