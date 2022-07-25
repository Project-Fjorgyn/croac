import numpy as np

from collections import defaultdict

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