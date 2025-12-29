import numpy as np
from pathlib import Path
import sounddevice as sd
import scipy.io.wavfile as wavfile

from keyword_asr import keyword_detect, transcribe
from speaker_verify import verify_user   # ← ADD THIS

# -------------------- CONFIG --------------------
SAMPLE_RATE = 16000
DURATION = 10
AUDIO_DIR = Path("enrolled_audio")
AUDIO_DIR.mkdir(exist_ok=True)
# ------------------------------------------------


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


def rms_check(audio, threshold=0.02):
    rms = np.sqrt(np.mean(audio**2))
    return rms >= threshold


def main():
    print("\n===== VOICE AUTH FLOW =====\n")

    audio = record_audio()

    # 1️⃣ Silence gate
    if not rms_check(audio):
        print("❌ Rejected: Silence / No speech")
        return

    verify_path = AUDIO_DIR / "verify.wav"
    save_wav(audio, verify_path)

    print("\n--- Speaker Verification ---")
    keyword = verify_user(verify_path)
    if not keyword:
        return


    # 2️⃣ Keyword / ASR gate
    transcript = transcribe(verify_path)
   
    if not keyword_detect(transcript, keyword):
        print("❌ Rejected: Keyword not found")
        print("Final Result: Not Authenticated")
        return

    print("✅ Keyword accepted")
    print("\nFinal Result: Authenticated ✅")


if __name__ == "__main__":
    main()
