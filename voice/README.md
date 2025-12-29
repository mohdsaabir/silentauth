# SilentAuth – Voice Iteration 2 

⚠️ **Status: Under Development**

This branch represents **Voice Iteration 2**, which introduces a redesigned
authentication pipeline with multi-layer verification.

## Current Focus
- Unified audio capture pipeline
- ASR-based speech & keyword validation
- Early rejection for non-speech audio
- Speaker verification as a second authentication layer

## Notes
- This branch contains **breaking changes**
- Code structure and APIs may change frequently
- Not intended for production or demo use yet

## Files and uses
- enroll.py records 5 audio clip and reads keyword and stores in DB
- keyword_asr.py is the module to check whether the keyword is present in the verifying audio or not
- speaker_verify.py this is to check the user identity using MFCC embeddings
- start_point.py this is the common file that connects the keyword_asr.py and speaker_verify.py

## Flow
- Run enroll.py for enrollment 
- Run start_point.py for verification

## Directory structure
<img width="331" height="592" alt="Screenshot 2025-12-29 221421" src="https://github.com/user-attachments/assets/024543a2-6a94-40d6-8f61-496e64406048" />

## Models 

- Download model files from the repo faster-whisper by comparing with directory structure model
- faster-whisper: https://huggingface.co/Systran/faster-whisper-base/tree/main
