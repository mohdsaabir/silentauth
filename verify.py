import sounddevice as sd
import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import scipy.io.wavfile as wavfile

# -------------------- CONFIG --------------------
SAMPLE_RATE = 16000
DURATION = 10  # seconds for verification
AUDIO_DIR = Path("enrolled_audio")
EMBED_DIR = Path("embeddings")
THRESHOLD = 0.75
# ------------------------------------------------

encoder = VoiceEncoder()

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds... Speak now 🎤")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return recording.squeeze()

def save_wav(audio, file_path):
    wavfile.write(file_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Audio saved to {file_path}")

def verify_user():
    enrolled_files = list(EMBED_DIR.glob("*.pt"))
    if not enrolled_files:
        print("No enrolled users found! Please enroll first.")
        return

    print("\n--- Verification ---")
    audio = record_audio()
    verify_path = AUDIO_DIR / "verify.wav"
    save_wav(audio, verify_path)

    wav_processed = preprocess_wav(str(verify_path))
    test_emb = encoder.embed_utterance(wav_processed).flatten()  # ensure 1D

    best_score = -1
    best_user = None

    for file in enrolled_files:
        user_id = file.stem
        stored_emb = torch.load(file).flatten().numpy()  # ensure 1D

        if stored_emb.shape != test_emb.shape:
            print(f"Skipping {user_id}: embedding shape mismatch {stored_emb.shape}")
            continue

        score = np.dot(test_emb, stored_emb) / (np.linalg.norm(test_emb) * np.linalg.norm(stored_emb))
        print(f"Similarity with {user_id}: {score*100:.2f}%")

        if score > best_score:
            best_score = score
            best_user = user_id

    print("\n--- RESULT ---")
    print(f"Best match: {best_user}")
    print(f"Score: {best_score*100:.2f}%")
    if best_score >= THRESHOLD:
        print("✅ AUTHENTICATED")
    else:
        print("❌ REJECTED")

if __name__ == "__main__":
    verify_user()
