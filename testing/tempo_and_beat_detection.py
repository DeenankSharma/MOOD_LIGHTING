import numpy as np
from matplotlib import pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import math
import glob
import os
import pandas as pd
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

test_case = {'src': './bh.wav', 'known_tempo': 126, 'start': 60.34, 'len': 5 }
test_case['end'] = test_case['start'] + test_case['len']
src = glob.glob(test_case['src'])[0]

y, sr = librosa.load(src, sr=48000, offset=test_case['start'], duration=test_case['len'])

ipd.display(pd.DataFrame([[sr, len(y), len(y.shape), np.max(y), np.min(y)]],columns=["Sample rate Hz", "Num Samples", "Channels", "Sample Max", "Sample Min"]).astype("int"))
sf.write('original_clip.wav', y, sr)

def predict_beats(samples, sr, hop_length=256):
    """I looked in the source code, I'm using 'onset_strength_multi' as it gave more options"""
    
    onset_env = librosa.onset.onset_strength_multi(
        y=samples,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,  # default is mean
        lag=1,                # default, unit? "time lag for computing differences"
        max_size=1,           # default, do not filter freq bins
        detrend=False,        # default, do not "filter onset strength to remove DC component"
        center=True,          # Centered frame analysis in STFT, by hop length
    )
    onset_env = onset_env[..., 0, :]
    
    # HOP_LENGTH = 512
    # onset_env = librosa.onset.onset_strength(y=samples, sr=sr,
    #                         # hop_length=hop_length,
    #                         aggregate=np.median, # default is mean
    #                         lag=1, # default, unit? "time lag for computing differences"
    #                         max_size=1, # default, do not filter freq bins
    #                         detrend=False, # default, do not "filter onset strength to remove DC component"
    #                         center=True, # Centered frame analysis in STFT, by hop length
    #                         )

    return librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        units="time",
        hop_length=hop_length,
        tightness=1000,            
        # start_bpm=126,
        # trim=False,
    )


reported_tempo, beats = predict_beats(y, sr)
expected_beats = math.floor(test_case["known_tempo"] * test_case["len"] / 60.0)

# Display the results
table = [
    ["Reported tempo", reported_tempo],
    ["Averaged tempo", 60 / np.average(np.diff(beats))],
    ["Num beats detected vs expected", f"{len(beats)} vs {expected_beats}"],
]
ipd.display(pd.DataFrame(table).astype("string"))

# Add in the click track from these detected beats
click_track = librosa.clicks(times=beats, sr=sr, length=len(y))
sf.write('output_with_clicks.wav', y + click_track, sr)

ipd.display(pd.DataFrame([
    ['Reported', reported_tempo],
    ['Averaged', 60 / np.average(np.diff(beats))],
    ['Min', 60 / np.max(np.diff(beats))],
    ['Max', 60 / np.min(np.diff(beats))],
    ['Median', 60 / np.median(np.diff(beats))],
    ['-','-'],
    ['Known', test_case['known_tempo']],
    ['Known seconds per beat', 60 / test_case['known_tempo'] ],
    ['Averaged seconds per beat', np.average(np.diff(beats))],
], columns=["Method", "BPM"]).astype("string"))

duration = 2.0 * test_case['len']
future, _ = librosa.load(src, sr=sr, offset=test_case['start'], duration=duration)

future_beats = np.array(beats)
diffs = np.diff(beats)
beats_added = 0

while future_beats[-1] < duration:
    future_beats = np.append(future_beats, future_beats[-1] + diffs[beats_added % len(diffs)])
    beats_added = beats_added + 1

future_click = librosa.clicks(times=future_beats, sr=sr, length=len(future))
ipd.Audio(future + future_click, rate=sr)