# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt

# # Loads an audio file as a waveform `y` which is in time series format and its sampling rate `sr`
# y, sr = librosa.load('./audiomass-output.wav')

# # Set parameters
# n_fft = 2048 # FFT window size
# hop_length = 512 # Hop length
# win_length = 2048 # Window length

# # Compute STFT
# stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

# # Get frequency bins 
# frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# # Calculate the magnitude spectrum i.e. the absolute value of the STFT result
# magnitude_spectrum = np.abs(stft_result)

# print(magnitude_spectrum)


import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import numpy as np
import tqdm
import plotly.io as pio

pio.kaleido.scope.default_format = "png"

AUDIO_FILE = './audiomass-output.wav'
FPS = 30
FFT_WINDOW_SECONDS = 0.25 # how many seconds of audio make up an FFT window

# Note range to display
FREQ_MIN = 10
FREQ_MAX = 8000

# Notes to display
TOP_NOTES = 3

# Names of the notes
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Output size. Generally use SCALE for higher res, unless you need a non-standard aspect ratio.
RESOLUTION = (1920, 1080)
SCALE = 2 # 0.5=QHD(960x540), 1=HD(1920x1080), 2=4K(3840x2160)
# import wave
# with wave.open('./audiomass-output.wav', 'rb') as wf:
#     print("Channels:", wf.getnchannels())
#     print("Sample width:", wf.getsampwidth())
#     print("Frame rate:", wf.getframerate())
#     print("Frame count:", wf.getnframes())

import soundfile as sf

audio, fs = sf.read('audiomass-output.wav')



FRAME_STEP = fs/FPS
FFT_WINDOW_SIZE = int(fs * FFT_WINDOW_SECONDS)
AUDIO_LENGTH = len(audio)/fs

def plot_fft(p, xf, fs, notes, dimensions=(960,540)):
  layout = go.Layout(
      title="frequency spectrum",
      autosize=False,
      width=dimensions[0],
      height=dimensions[1],
      xaxis_title="Frequency (note)",
      yaxis_title="Magnitude",
      font={'size' : 24}
  )

  fig = go.Figure(layout=layout,
                  layout_xaxis_range=[FREQ_MIN,FREQ_MAX],
                  layout_yaxis_range=[0,1]
                  )
  
  fig.add_trace(go.Scatter(
      x = xf,
      y = p))
  
  for note in notes:
    fig.add_annotation(x=note[0]+10, y=note[2],
            text=note[1],
            font = {'size' : 48},
            showarrow=False)
  return fig

def extract_sample(audio, frame_number):
  end = frame_number * FRAME_OFFSET
  begin = int(end - FFT_WINDOW_SIZE)

  if end == 0:
    # We have no audio yet, return all zeros (very beginning)
    print("Beginning of audio, returning zeros")
    return np.zeros((np.abs(begin)),dtype=float)
  elif begin<0:
    # We have some audio, padd with zeros
    print("Beginning of audio, padding with zeros")
    return np.concatenate([np.zeros((np.abs(begin)),dtype=float),audio[0:end]])
  else:
    # Usually this happens, return the next sample
    
    print(f"Extracting sample from {begin} to {end}")
    return audio[begin:end]
  
  
def find_top_notes(fft,num):
  if np.max(fft.real)<0.001:
    return []

  lst = [x for x in enumerate(fft.real)]
  lst = sorted(lst, key=lambda x: x[1],reverse=True)

  idx = 0
  found = []
  found_note = set()
  while( (idx<len(lst)) and (len(found)<num) ):
    f = xf[lst[idx][0]]
    y = lst[idx][1]
    n = freq_to_number(f)
    n0 = int(round(n))
    name = note_name(n0)

    if name not in found_note:
      found_note.add(name)
      s = [f,note_name(n0),y]
      found.append(s)
    idx += 1
    
  return found

def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): return NOTE_NAMES[n % 12] + str(int(n/12 - 1))

# Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, FFT_WINDOW_SIZE, False)))

xf = np.fft.rfftfreq(FFT_WINDOW_SIZE, 1/fs)
FRAME_COUNT = int(AUDIO_LENGTH*FPS)
FRAME_OFFSET = int(len(audio)/FRAME_COUNT)

def plot_fft_matplotlib(p, xf, notes, dimensions=(1920, 1080), frame_number=0):
    plt.figure(figsize=(dimensions[0]/100, dimensions[1]/100), dpi=100)
    plt.plot(xf, p, color='blue')
    plt.xlim(FREQ_MIN, FREQ_MAX)
    plt.ylim(0, 1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"Frequency Spectrum - Frame {frame_number}")
    
    # Add note annotations
    for note in notes:
        freq, name, amp = note
        plt.text(freq, amp + 0.02, name, fontsize=16, color='red', ha='center')
    
    plt.tight_layout()
    os.makedirs('./contents', exist_ok=True)
    plt.savefig(f'./contents/frame{frame_number:04d}.png')
    plt.close()

# Pass 1, find out the maximum amplitude so we can scale.
mx = 0
for frame_number in range(FRAME_COUNT):
  sample = extract_sample(audio, frame_number)

  fft = np.fft.rfft(sample * window)
  fft = np.abs(fft).real 
  
  mx = max(np.max(fft),mx)

print(f"Max amplitude: {mx}")

# Pass 2, produce the animation
for frame_number in tqdm.tqdm(range(FRAME_COUNT)):
  sample = extract_sample(audio, frame_number)
  print(f"Processing frame {frame_number} of {FRAME_COUNT}...")
  fft = np.fft.rfft(sample * window)
  fft = np.abs(fft) / mx 
     
  s = find_top_notes(fft,TOP_NOTES)

  plot_fft_matplotlib(fft.real, xf, s, RESOLUTION, frame_number)

