import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import scipy.io.wavfile as wavfile
from db_utils import fetch_all_embeddings 

# -------------------- CONFIG --------------------
SAMPLE_RATE = 16000
DURATION = 10
THRESHOLD = 0.78
AUDIO_DIR = Path("enrolled_audio")
AUDIO_DIR.mkdir(exist_ok=True)
# ------------------------------------------------

encoder = VoiceEncoder()


def l2_normalize(x, eps=1e-10):
    return x / (np.linalg.norm(x) + eps)


'''

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds... Speak now 🎤")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return recording.squeeze()


def save_wav(audio, file_path):
    wavfile.write(file_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Audio saved to {file_path}")

'''

def verify_user(verify_path):
    enrolled = fetch_all_embeddings()  # returns list of (user_name, embedding) from db_utils.py
    if not enrolled:
        print("No enrolled users found in the database!")
        return

    print("\n--- Verification ---")
    '''
    audio = record_audio()


    # Stores the audio clip locally can remove this code later in production 
    verify_path = AUDIO_DIR / "verify.wav"
    save_wav(audio, verify_path)
    '''

    wav_processed = preprocess_wav(str(verify_path))
    test_emb = encoder.embed_utterance(wav_processed).flatten()
    test_emb = l2_normalize(test_emb)

    best_score = -1
    best_user = None
    keyword= None
    for user_name, stored_emb, keywords in enrolled:
        stored_emb = l2_normalize(stored_emb)  # normalize db embeddings
        if stored_emb.shape != test_emb.shape:
            print(f"Skipping {user_name}: shape mismatch {stored_emb.shape}")
            continue

        score = np.dot(test_emb, stored_emb)  # cosine similarity

        #print(f"Similarity with {user_name}: {score*100:.2f}%")

        if score > best_score:
            best_score = score
            best_user = user_name
            keyword = keywords

    print("\n--- RESULT ---")
    print(f"Best match: {best_user}")
    print(f"Score: {best_score*100:.2f}%")
    if best_score >= THRESHOLD:
        print("✅ AUTHENTICATED")
        print(f"Keyword: {keyword}")
        return keyword
    else:
        print("❌ REJECTED")
        return False


if __name__ == "__main__":
    verify_user()
