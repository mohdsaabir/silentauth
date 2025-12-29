import sounddevice as sd
import numpy as np
import soundfile as sf
from pathlib import Path
import re
from faster_whisper import WhisperModel

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 10  # seconds
AUDIO_DIR = Path("asr_audio")
AUDIO_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path("model/base")   # Faster-Whisper base model
DEVICE = "cpu"
LANGUAGE = "en"

KEYWORDS = [
    "hello system",
    "i am",
    "muhammed sabir"
    "verify me"
]
# ----------------------------------------

'''
def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds... Speak clearly 🎤")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return recording.squeeze()


def save_wav(audio, file_path):
    sf.write(file_path, audio, SAMPLE_RATE)
    print(f"Audio saved to {file_path}")
'''



def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_detect(transcript, keyword):
    transcript = normalize_text(transcript)
    print("\nNormalized text:", transcript)

    
    if keyword in transcript:
        print(f"\n✅ KEYWORD DETECTED → '{keyword}'")
        return True

    print("\n❌ NO KEYWORD DETECTED")
    return False


def transcribe(audio_path):
    print("\nLoading Faster Whisper model...")
    model = WhisperModel(
        str(MODEL_PATH),
        device=DEVICE,
        compute_type="int8"  # faster on CPU
    )

    #print("Transcribing...")
    segments, info = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        beam_size=1,
        vad_filter=True
    )

    print("\n===== ASR OUTPUT =====")
    #print(f"Detected language: {info.language}")

    full_text = ""
    for segment in segments:
        #print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "

    full_text = full_text.strip()
    #print("\nFull transcription:", full_text)

    return full_text



'''
def main():
    audio = record_audio()
    wav_path = AUDIO_DIR / "recorded.wav"
    save_wav(audio, wav_path)

    transcript = transcribe(wav_path)

    print("\n===== FINAL RESULT =====")
    if keyword_detect(transcript):
        print("✅ ACCEPTED")
    else:
        print("❌ REJECTED")


if __name__ == "__main__":
    main()
'''