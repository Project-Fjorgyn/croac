import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

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
        self.delay_bounds = int(np.floor(np.min(self.delays))), int(np.ceil(np.max(self.delays)))

    def ingest_signals(self, signals):
        self.signals = signals
        self.samples = np.arange(0, signals.shape[1], 1)
        self.allowable_samples = self.samples[-self.delay_bounds[0]:-self.delay_bounds[1]]
        self.interpolators = [
            interp1d(
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

    def decompose(self):
        rows = []
        for i in range(self.result.shape[0]):
            row = []
            for j in range(self.result.shape[1]):
                signal = self.result[i,j,:]
                N = signal.shape[0]
                m = int(np.floor(N/2)) # we can't use things above nyquist
                amp = (np.abs(np.fft.fft(signal))/N)[:m]
                row.append(amp)
            rows.append(row)
        self.decomposition = np.array(rows)

    def filter_frequencies(self, percentile=0.95):
        # we sum over the angles
        overall_impact = pd.DataFrame(self.decomposition.sum(axis=(0,1)), columns=['impact']).reset_index()
        overall_impact = overall_impact.sort_values('impact', ascending=False)
        overall_impact['cumulative'] = overall_impact['impact'].cumsum()/overall_impact['impact'].sum()
        filter_value = overall_impact[overall_impact['cumulative'] > percentile]['impact'].values[0]
        indices_to_zero = overall_impact[overall_impact['impact'] < filter_value]['index']
        self.filtered_decomposition = self.decomposition.copy()
        for i in indices_to_zero:
            self.filtered_decomposition[:,:,i] = 0.
