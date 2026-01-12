# SilentAuth – Voice Authentication Module (Iteration 3)

This module implements **speaker enrollment and verification** using a
state-of-the-art **ECAPA-TDNN** speaker recognition model via SpeechBrain.

It is designed as an **offline-capable biometric authentication component**
for the SilentAuth multimodal system.

---

## Overview

The voice module supports:

- **Voice enrollment (registration)** of users
- **Voice verification (authentication)** against enrolled users
- **SQLite-based embedding storage**
- **Cosine similarity–based decision logic**

The system focuses on **speaker identity**, not speech content.

---

## Model Used

- **ECAPA-TDNN** (SpeechBrain)
- Trained on **VoxCeleb**
- Produces **192-dimensional speaker embeddings**
- Robust to noise, language, and spoken content

---

## Files in This Module

```text
voice_module/
├── ecapa_realtime_register.py      # Voice enrollment script
├── ecapa_realtime_verify_db.py     # Voice verification script
├── requirements.txt                # Python dependencies
└── README.md

## Requirements

Python 3.10.0

## Usage

Install dependencies using:
```bash
pip install -r requirements.txt
```
Register a user:
```bash
python ecapa_realtime_register.py
```
Verify a user:
```bash
python ecapa_realtime_verify_db.py  
```