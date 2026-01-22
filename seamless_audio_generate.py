import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

INPUT_DIR = "static/vehicle_sounds_raw"
OUTPUT_DIR = "static/vehicle_sounds"
REPEAT_COUNT = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def trim_silence(audio, silence_thresh=-45, chunk_size=5):
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=chunk_size,
        silence_thresh=silence_thresh
    )

    if not nonsilent_ranges:
        return audio

    start = nonsilent_ranges[0][0]
    end = nonsilent_ranges[-1][1]
    return audio[start:end]

for filename in os.listdir(INPUT_DIR):
    if filename.startswith("car_") and filename.endswith(".mp3"):
        input_path = os.path.join(INPUT_DIR, filename)

        audio = AudioSegment.from_file(input_path, format="mp3")

        # Remove spatial/stereo effect
        audio = audio.set_channels(1)

        # Trim silence
        audio = trim_silence(audio)

        stitched_audio = audio * REPEAT_COUNT

        name, _ = os.path.splitext(filename)
        output_filename = f"{name}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        stitched_audio.export(output_path, format="wav")

        print(f"Created: {output_filename}")
