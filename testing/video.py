import subprocess

FPS = 30
AUDIO_FILE = "audiomass-output.wav"

subprocess.run([
    "ffmpeg", "-y",
    "-r", str(FPS),
    "-f", "image2",
    "-s", "1920x1080",
    "-i", "contents/frame%04d.png",
    "-i", AUDIO_FILE,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "movie.mp4"
])
