import numpy as np
from pathlib import Path
import sounddevice as sd
import scipy.io.wavfile as wavfile

# These are the modules we created
from keyword_asr import keyword_detect, transcribe
from speaker_verify import verify_user   

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

    # Checks for silence using RMS
    if not rms_check(audio):
        print("❌ Rejected: Silence / No speech")
        return

    verify_path = AUDIO_DIR / "verify.wav"
    save_wav(audio, verify_path)

    # Verification using speaker embedding for user identification
    result = None
    #print("\n---This is print is for testing---")
    print("\n--- Speaker Verification ---")
    keyword, result = verify_user(verify_path)
    if not keyword:
        # This is the return if speaker verification fails it doesnot proceed to ASR 
        print(f"\nAuthentication Details: {result}")
        return


    # ASR and Keyword Detection
    transcript = transcribe(verify_path)
   
    if not keyword_detect(transcript, keyword):
        #print("❌ Rejected: Keyword not found")
        #print("Final Result: Not Authenticated")
        result = {"modality":"Voice", "user_name": None, "similarity_score": 0, "status":"Rejected"}
        # This return if the speaker is verified but keyword is not detected so similarity score is 0 
        print(f"\nAuthentication Details: {result}")
        return

    #print("✅ Keyword accepted")
    #print("\nFinal Result: Authenticated ✅")
    # This should return the authentication details while integration with other modalities
    print(f"\nAuthentication Details: {result}")


if __name__ == "__main__":
    main()
