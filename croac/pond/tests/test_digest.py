import unittest
import numpy as np
import pandas as pd

from pandas.testing import assert_frame_equal

from ..digest import (
    get_sample_size_params,
    get_base_frequencies,
    build_fft_dataframe,
    remove_noisy_frequencies,
    remove_silent_sections,
    note_generator
)

class TestGetSampleSizeParams(unittest.TestCase):
    def test_make_N_integer(self):
        sample_rate = 1000
        freq_resolution = 44.1
        expected = 22, 11, 22/1000
        actual = get_sample_size_params(sample_rate, freq_resolution)
        assert expected == actual

    def test_make_m_integer(self):
        sample_rate = 1000
        freq_resolution = 90.9
        expected = 11, 5, 11/1000
        actual = get_sample_size_params(sample_rate, freq_resolution)
        assert expected == actual

class TestGetBaseFrequencies(unittest.TestCase):
    def test_base(self):
        sample_time = 0.22
        m = 3
        expected = np.array([0/0.22, 1/0.22, 2/0.22])
        actual = get_base_frequencies(m, sample_time)
        assert (expected == actual).all()

class TestBuildFFTDataFrame(unittest.TestCase):
    def test_base(self):
        # for these choices N, m, sample_time = 20, 10, 0.02
        freq_resolution = 50
        sample_rate = 1000
        # setup enough time samples for 5 windows
        t = np.arange(0, 0.1, 1/1000)
        # we'll have two sine waves in the first 5 windows
        long_wave = 10 * np.sin(2*np.pi * (100 * t))
        short_wave = 5 * np.sin(2*np.pi * (250 * t))
        # and one in the second 5 windows
        wave = np.concatenate([long_wave + short_wave, long_wave])
        # now we build our dataframe
        df = build_fft_dataframe(wave, sample_rate, freq_resolution)
        for _, row in df.iterrows():
            time, amp, freq = row['time'], row['amplitude'], row['frequency']
            if freq == 250 and time < 0.1:
                assert abs(amp - 2.5) < 10 ** -10
            elif freq == 100:
                assert abs(amp - 5) < 10 ** -10
            else:
                assert abs(amp) < 10 ** -10
        assert tuple(df.shape) == (10 * 10, 3)

class TestRemoveNoisyFrequencies(unittest.TestCase):
    def test_base(self):
        fft_df = pd.DataFrame(
            [
                {'time': 0, 'amplitude': i, 'frequency': 9 - i}
                for i in range(10)
            ]
            +
            [
                {'time': 1, 'amplitude': 9 - i, 'frequency': i}
                for i in range(10)
            ]
        )
        percentile = 0.8
        expected = pd.DataFrame(
            [
                {'time': 0, 'amplitude': i, 'frequency': 9 - i}
                for i in range(5, 10)
            ]
            +
            [
                {'time': 1, 'amplitude': 9 - i, 'frequency': i}
                for i in range(5)
            ]
        )
        actual = remove_noisy_frequencies(fft_df, percentile)
        assert_frame_equal(
            expected.sort_values(['time', 'amplitude']).reset_index(drop=True),
            actual.sort_values(['time', 'amplitude']).reset_index(drop=True),
        )

class TestRemoveSilentSections(unittest.TestCase):
    def test_base(self):
        fft_df = pd.DataFrame([
            {'time': 0, 'amplitude': 7.5, 'frequency': 100},
            {'time': 0, 'amplitude': 2.5, 'frequency': 50},
            {'time': 1, 'amplitude': 1, 'frequency': 100},
            {'time': 2, 'amplitude': 0.5, 'frequency': 20}
        ])
        cutoff, percentile = 0.1, 0.95
        expected = pd.DataFrame([
            {'time': 0, 'amplitude': 7.5, 'frequency': 100},
            {'time': 0, 'amplitude': 2.5, 'frequency': 50},
            {'time': 1, 'amplitude': 1, 'frequency': 100}
        ])
        actual = remove_silent_sections(fft_df, cutoff, percentile)
        assert_frame_equal(
            expected.sort_values(['time', 'amplitude']).reset_index(drop=True),
            actual.sort_values(['time', 'amplitude']).reset_index(drop=True),
        )

class TestNoteGenerator(unittest.TestCase):
    def test_base(self):
        fft_df = pd.DataFrame([
            {'time': 0, 'amplitude': 7.5, 'frequency': 100},
            {'time': 0, 'amplitude': 2.5, 'frequency': 50},
            {'time': 1, 'amplitude': 1, 'frequency': 100},
            {'time': 3, 'amplitude': 0.5, 'frequency': 20}
        ])
        sample_time = 1
        note1, note2 = tuple(note_generator(fft_df, sample_time))

        assert note1.sample_time == sample_time
        assert len(note1.definition) == 2
        assert note1.definition[0] == {
            100: 7.5,
            50: 2.5
        }
        assert note1.definition[1] == {
            100: 1
        }

        assert note2.sample_time == sample_time
        assert len(note2.definition) == 1
        assert note2.definition[0] == {
            20: 0.5
        }