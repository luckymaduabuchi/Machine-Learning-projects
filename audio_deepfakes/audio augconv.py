from pydub import AudioSegment
import os

def mp3_to_wav(input_folder, duration=5, sample_rate=32000):
    try:
        # Iterate through each file in the input directory
        for filename in os.listdir(input_folder):
            if filename.endswith(".mp3"):
                # Load the MP3 file
                mp3_file_path = os.path.join(input_folder, filename)
                audio = AudioSegment.from_mp3(mp3_file_path)
                
                # Set the duration and sample rate
                audio = audio[:duration * 1000]  # Convert to milliseconds
                audio = audio.set_frame_rate(sample_rate)
                
                # Set the output file path
                output_file_path = os.path.splitext(mp3_file_path)[0] + ".wav"
                
                # Export the audio to a WAV file
                audio.export(output_file_path, format="wav")
                
                # Delete the original MP3 file
                os.remove(mp3_file_path)
                
                print(f"Conversion successful. WAV file saved at {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Path to the input directory containing MP3 files
    input_folder = "/home/vm-user/Downloads/Music /Music /audio/VoiceT/AUG "

    # Convert all MP3 files in the input directory to WAV
    mp3_to_wav(input_folder)
