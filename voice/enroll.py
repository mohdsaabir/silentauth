import sounddevice as sd
import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import scipy.io.wavfile as wavfile
import re

from db_utils import insert_embedding

# -------------------- CONFIG --------------------
SAMPLE_RATE = 16000
DURATION = 10        
NUM_SAMPLES = 5
SILENCE_RMS_THRESHOLD = 0.02

AUDIO_DIR = Path("enrolled_audio")
EMBED_DIR = Path("embeddings")
AUDIO_DIR.mkdir(exist_ok=True)
EMBED_DIR.mkdir(exist_ok=True)
# ------------------------------------------------

encoder = VoiceEncoder()


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)

# Silence detection using RMS
def is_speech(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"RMS: {rms:.4f}")
    return rms >= SILENCE_RMS_THRESHOLD


def normalize_keyword(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds... Speak now 🎤")
    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return recording.squeeze()


def save_wav(audio, file_path):
    wavfile.write(
        file_path,
        SAMPLE_RATE,
        (audio * 32767).astype(np.int16)
    )
    print(f"Audio saved to {file_path}")


def enroll_user(user_id):
    embeddings = []

    print(f"\n--- Enrollment for user '{user_id}' ---")

    # ---------------- KEYWORD ENROLLMENT ----------------
    raw_keyword = input(
        "\nEnter a secret keyword (Should not include your name or related terms): "
    )
    keyword = normalize_keyword(raw_keyword)

   
    print(f"Keyword saved: '{keyword}'")

    # ---------------- VOICE ENROLLMENT ----------------
    sample_count = 0

    while sample_count < NUM_SAMPLES:
        print(f"\nSample {sample_count + 1}/{NUM_SAMPLES}")
        audio = record_audio()

        if not is_speech(audio):
            print("❌ Silence detected. Please speak clearly. Retrying...")
            continue

        audio_path = AUDIO_DIR / f"{user_id}_sample{sample_count + 1}.wav"
        save_wav(audio, audio_path)

        wav_processed = preprocess_wav(str(audio_path))
        emb = encoder.embed_utterance(wav_processed).flatten()
        emb = l2_normalize(emb)

        embeddings.append(emb)
        sample_count += 1

    # ---------------- FINAL EMBEDDING ----------------
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = l2_normalize(avg_emb)

    emb_path = EMBED_DIR / f"{user_id}.pt"
    torch.save(torch.tensor(avg_emb, dtype=torch.float32), emb_path)

    insert_embedding(user_id, avg_emb, keyword)

    print(f"\n✅ Enrollment complete for '{user_id}'")
    print(f"Embedding saved to {emb_path}\n")


if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    enroll_user(user_id)
