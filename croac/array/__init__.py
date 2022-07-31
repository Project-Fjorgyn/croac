import scipy
import numpy as np

SPEED_OF_SOUND = 343 # m/s

class DelayArray(object):
    def __init__(self, antenna_x, antenna_y, theta, phi, sample_rate):
        self.theta, self.phi, self.antenna_x = np.meshgrid(theta, phi, antenna_x, indexing='ij')
        *_, self.antenna_y = np.meshgrid(theta, phi, antenna_y, indexing='ij')
        self.sample_rate = sample_rate
        self._compute_delays()

    def _compute_delays(self):
        self.delays = - self.antenna_x * np.sin(self.theta) * np.cos(self.phi)
        self.delays -= self.antenna_y * np.sin(self.theta) * np.sin(self.phi)
        self.delays *= self.sample_rate / SPEED_OF_SOUND
        self.delay_bounds = np.floor(np.min(self.delays)), np.ceil(np.max(self.delays))

    def ingest_signals(self, signals):
        self.signals = signals
        self.samples = np.arange(0, signals.shape[0], 1)
        self.allowable_samples = np.samples[-self.delay_bounds[0]:-self.delay_bounds[1]]
        self.interpolators = [
            scipy.interpolate.interp1d(
                self.samples, signal, kind='linear'
            )
            for signal in self.signals
        ]

    def sweep(self):
        rows = []
        for i in range(self.delays.shape[0]):
            row = []
            for j in range(self.delays.shape[1]):
                delays = self.delays[i,j,:]
                delayed_signals = np.array([
                    interpolator(self.allowable_samples + delay)
                    for delay, interpolator in zip(delays, self.interpolators)
                ])
                result = delayed_signals.sum(axis=0)
                row.append(result)
            rows.append(row)
        self.result = np.array(rows)


