import os
import string

def generate_names():
    letters = string.ascii_uppercase
    for letter1 in letters:
        yield letter1
    for letter1 in letters:
        for letter2 in letters:
            yield letter1 + letter2

def rename_audio_files(folder_path):
    names_generator = generate_names()
    files = sorted(os.listdir(folder_path))
    for file in files:
        if file.endswith('.mp3') or file.endswith('.wav') or file.endswith('.ogg'):
            new_name = next(names_generator) + os.path.splitext(file)[1]
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
            print(f"Renamed {file} to {new_name}")

# Change the path to your audio files folder
folder_path = '/home/vm-user/audio/Voice changer t/'

rename_audio_files(folder_path)
