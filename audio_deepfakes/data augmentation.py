from pydub import AudioSegment
import os

# Function to perform pitch shifting
def pitch_shift(audio, semitones):
    return audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * (2 ** (semitones / 12.0)))
    })

# Function to perform speed change
def speed_change(audio, speed_factor):
    return audio.speedup(playback_speed=speed_factor)

# Function to perform amplitude scaling
def amplitude_scale(audio, db_gain):
    return audio + db_gain

# Function to perform time shifting
def time_shift(audio, ms):
    return audio._spawn(b"\x00" * int(ms * audio.frame_rate * 2) + audio.raw_data)

# Path to the directory containing audio files
input_folder = "/home/vm-user/Downloads/Music /Music /audio/VoiceT/Text "
output_folder = "/home/vm-user/Downloads/Music /Music /audio/VoiceT/AUG"

print(os.listdir(input_folder))

# Iterate through each audio file in the input directory
for file_name in os.listdir(input_folder):
    if file_name.endswith(".mp3"):
        file_path = os.path.join(input_folder, file_name)
        audio = AudioSegment.from_mp3(file_path)

        # Perform data augmentation
        augmented_audio = pitch_shift(audio, semitones=2)
        augmented_audio = speed_change(augmented_audio, speed_factor=0.9)
        augmented_audio = amplitude_scale(augmented_audio, db_gain=-10)
        augmented_audio = time_shift(augmented_audio, ms=500)

        # Save augmented audio to output directory
        output_file_path = os.path.join(output_folder, f"augmented_{file_name}")
        augmented_audio.export(output_file_path, format="mp3")
