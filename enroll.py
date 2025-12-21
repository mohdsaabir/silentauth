import sounddevice as sd
import numpy as np
import torch
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import scipy.io.wavfile as wavfile
from db_utils import insert_embedding

# -------------------- CONFIG --------------------
SAMPLE_RATE = 16000
DURATION = 10        # seconds per sample
NUM_SAMPLES = 5
AUDIO_DIR = Path("enrolled_audio")
EMBED_DIR = Path("embeddings")
AUDIO_DIR.mkdir(exist_ok=True)
EMBED_DIR.mkdir(exist_ok=True)
# ------------------------------------------------

encoder = VoiceEncoder()


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)

# Records audio from microphone converts the 2D array to 1D array using squeeze
def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds... Speak now 🎤")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return recording.squeeze()

# Saves the recorded audio to a WAV file
def save_wav(audio, file_path):
    wavfile.write(file_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Audio saved to {file_path}")


def enroll_user(user_id):
    embeddings = []
    print(f"\n--- Enrollment for user '{user_id}' ---")

    for i in range(NUM_SAMPLES):
        audio = record_audio()
        audio_path = AUDIO_DIR / f"{user_id}_sample{i+1}.wav"
        save_wav(audio, audio_path)

        wav_processed = preprocess_wav(str(audio_path))
        emb = encoder.embed_utterance(wav_processed).flatten()
        emb = l2_normalize(emb)           
        embeddings.append(emb)

    # Average + normalize again
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = l2_normalize(avg_emb)

    # Saving the embedding in both local system and as well as DB
    emb_path = EMBED_DIR / f"{user_id}.pt"
    torch.save(torch.tensor(avg_emb, dtype=torch.float32), emb_path)
    insert_embedding(user_id, avg_emb)

    print(f"\nEnrollment complete for {user_id}")
    print(f"Embedding saved to {emb_path}\n")


if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    enroll_user(user_id)
