import soundfile as sf
import numpy as np
import pandas as pd
import seaborn as sns
import wave

from tqdm import tqdm

from collections import defaultdict

def find_cumulative_boundary(a, boundary):
    a = -np.sort(-a)
    c = a.cumsum()/a.sum()
    return a[c <= boundary].min()

def digest(data, resolution=44.1, samplerate=44100, boundary=0.8, cutoff=0.1):
    N = int(np.floor(samplerate/resolution))
    mid = int(np.floor(N/2))
    sampletime = N/samplerate
    base_freq = np.arange(0, mid, 1)/sampletime
    rows = []
    for s in tqdm(range(0, data.shape[0], N)):
        t = s/samplerate
        X = data[s:s+N]
        amp = (np.abs(np.fft.fft(X))/N)[:mid]
        a_bound = find_cumulative_boundary(amp, boundary)
        freq = base_freq[amp >= a_bound]
        amp = amp[amp >= a_bound]
        for a, f in zip(amp, freq):
            rows.append({
                'time': t,
                'amplitude': a,
                'frequency': f
            })
    df = pd.DataFrame(rows)
    summed = df.groupby('time').sum()[['amplitude']].reset_index()
    filtered = summed[summed['amplitude'] >= cutoff * summed['amplitude'].quantile(0.95)]
    df = df.merge(filtered[['time']], on='time', how='inner')
    df['amplitude'] /= filtered['amplitude'].quantile(0.95)
    last_time = -float('inf')
    offset = 0
    current_note = None
    for _, row in df.sort_values('time').iterrows():
        time = row['time']
        if last_time < time:
            interval = time - last_time
            if abs(interval - sampletime) > sampletime * 10 ** -3:
                if current_note is not None:
                    yield current_note
                current_note = Note()
                current_note.sampletime = sampletime
                offset = time
            current_note.definition.append({})
        current_note.definition[-1][row['frequency']] = row['amplitude']
        last_time = time
    yield current_note

class Note(object):
    def __init__(self):
        self.definition = []
        self.phases = defaultdict(lambda: np.random.rand())
        
    def listen(self, samplerate=44100, pitch=1., duration=1., amplitude=1.):
        vs = np.array([])
        for i, info in enumerate(self.definition):
            sampletime = self.sampletime * duration
            N = round(samplerate * sampletime)
            t = np.arange(i, i + N, 1)*sampletime/N
            v = np.array([1j - 1j] * t.shape[0])
            for freq, amp in info.items():
                v += amp * np.exp(2j * np.pi * (pitch * freq * t + self.phases[freq]))
            vs = np.concatenate([vs, v])
        return np.real(vs)
    
class Phrase(object):
    def __init__(self, notes, pauses, shapes):
        self.notes = notes
        self.pauses = pauses
        self.shapes = shapes
        
    def listen(self, samplerate=44100, pitch=1., duration=1., amplitude=1.):
        sound = np.array([])
        for note, pause, shape in zip(self.notes, self.pauses, self.shapes):
            sound = np.concatenate([
                sound, 
                note.listen(
                    samplerate, 
                    shape.get('pitch', 1) * pitch, 
                    shape.get('duration', 1) * duration, 
                    shape.get('amplitude', 1) * amplitude), 
                np.zeros(round(pause * samplerate))])
        return sound