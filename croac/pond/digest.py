import numpy as np
import pandas as pd

from .note import Note

def get_sample_size_params(sample_rate, freq_resolution):
    N = int(np.floor(sample_rate/freq_resolution))
    m = int(np.floor(N/2))
    sample_time = N/sample_rate
    return N, m, sample_time

def get_base_frequencies(m, sample_time):
    return np.arange(0, m, 1) / sample_time

def build_fft_dataframe(data, sample_rate, freq_resolution):
    N, m, sample_time = get_sample_size_params(sample_rate, freq_resolution)
    base_freq = get_base_frequencies(m, sample_time)
    rows = []
    for s in range(0, data.shape[0], N):
        t = s/sample_rate
        X = data[s:s+N]
        amp = (np.abs(np.fft.fft(X))/N)[:m]
        for a, f in zip(amp, base_freq):
            rows.append({
                'time': t,
                'amplitude': a,
                'frequency': f
            })
    return pd.DataFrame(rows)

def remove_noisy_frequencies(fft_df, percentile):
    dfs = []
    for time in sorted(fft_df['time'].unique()):
        df = fft_df[fft_df['time'] == time].sort_values('amplitude', ascending=False)
        if df['amplitude'].sum() == 0:
            dfs.append(df)
            continue
        df['amplitude_percentile'] = df['amplitude'].cumsum() / df['amplitude'].sum()
        boundary = df[df['amplitude_percentile'] > percentile]['amplitude'].values[0]
        del df['amplitude_percentile']
        dfs.append(df[df['amplitude'] > boundary])
    return pd.concat(dfs)

def remove_silent_sections(fft_df, cutoff, percentile):
    summed = fft_df.groupby('time').sum()[['amplitude']].reset_index()
    filtered = summed[summed['amplitude'] > cutoff * summed['amplitude'].quantile(percentile)]
    return fft_df.merge(filtered[['time']], on='time', how='inner')

def note_generator(fft_df, sample_time):
    last_time = -float('inf')
    current_note = None
    for _, row in fft_df.sort_values('time').iterrows():
        time = row['time']
        if last_time < time:
            interval = time - last_time
            if interval - sample_time > 10 ** -10:
                # in this case we just skipped a silent
                # section so a new note is required
                if current_note is not None:
                    yield current_note, interval - sample_time
                current_note = Note()
                current_note.sample_time = sample_time
            current_note.definition.append({})
        current_note.definition[-1][row['frequency']] = row['amplitude']
        last_time = time
    yield current_note, 0

def digest(data, sample_rate=44100, freq_resolution=44.1, noise_percentile=0.8, cutoff=0.1, silence_percentile=0.95):
    fft_df = build_fft_dataframe(data, sample_rate, freq_resolution)
    fft_df = remove_noisy_frequencies(fft_df, noise_percentile)
    fft_df = remove_silent_sections(fft_df, cutoff, silence_percentile)
    *_, sample_time = get_sample_size_params(sample_rate, freq_resolution)
    for note, pause in note_generator(fft_df, sample_time):
        yield note, pause
