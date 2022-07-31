import numpy as np

from collections import defaultdict

class Note(object):
    def __init__(self):
        self.definition = []
        self.phases = defaultdict(np.random.rand)
        
    def listen(self, sample_rate=44100, pitch=1., duration=1., amplitude=1.):
        vs = np.array([])
        for i, info in enumerate(self.definition):
            sample_time = self.sample_time * duration
            N = round(sample_rate * sample_time)
            t = np.arange(i * N, (i + 1) * N, 1)*sample_time/N
            v = np.array([1j - 1j] * t.shape[0])
            for freq, amp in info.items():
                v += amplitude * amp * np.exp(2j * np.pi * (pitch * freq * t + self.phases[freq]))
            vs = np.concatenate([vs, v])
        return np.real(vs)
