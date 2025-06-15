import wave         # Used to open and read WAV audio files
import pyaudio      # Used to play audio through your system's speakers

def play_wav_file(path):
    CHUNK = 1024  # 👈 Number of audio frames to read at once (like reading a few words at a time from a long book)

    try:
        # 👇 Open the wave file in read-binary mode
        with wave.open(path, 'rb') as wf:
            # 👇 Create a PyAudio instance (connects Python to your audio hardware)
            p = pyaudio.PyAudio()

            # 👇 Open an output audio stream (like opening a pipe to send sound to speakers)
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),  # e.g., 8-bit or 16-bit audio
                            channels=wf.getnchannels(),                        # Mono = 1, Stereo = 2
                            rate=wf.getframerate(),                            # Sample rate: e.g., 44100 samples per second
                            output=True)                                      # We're only playing audio (not recording)

            # 👇 Read and play the file in small chunks
            while len(data := wf.readframes(CHUNK)):  # 'data' holds a small portion of sound each time
                stream.write(data)                    # Send that chunk of audio to the speakers

            # 👇 Close the audio stream after playback is done
            stream.close()
            p.terminate()  # Properly shut down the PyAudio system

            print("Playback finished.")

    # 👇 Error handling for common problems
    except FileNotFoundError:
        print(f"File not found: {path}")
    except wave.Error:
        print(f"Invalid WAV file: {path}")
    except Exception as e:
        print(f"Error: {e}")
