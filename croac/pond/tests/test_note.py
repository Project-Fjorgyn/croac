import unittest
import numpy as np

from collections import defaultdict

from ..note import (
    Note
)

class TestNoteListen(unittest.TestCase):
    def setUp(self):
        self.note = Note()
        self.note.definition = [
            {
                100: 10,
                250: 2,
                400: 5
            },
            {
                100: 7,
                250: 2,
                300: 2
            }
        ]
        self.note.sample_time = 0.1

    def test_sample_rate(self):
        sample_rate = 100
        results = self.note.listen(sample_rate, 1., 1., 1.)
        assert results.shape[0] == \
            len(self.note.definition) * self.note.sample_time * sample_rate

    def test_pitch(self):
        # we're going to set the phases so we know what to expect
        self.note.phases = defaultdict(lambda: 0.5)
        sample_rate = 100
        pitch = 1.1

        N = sample_rate * self.note.sample_time
        t1 = np.arange(0, N, 1) / sample_rate
        v1 = (
            10 * np.exp(2j * np.pi * (100 * t1 * pitch + 0.5)) 
            + 2 * np.exp(2j * np.pi * (250 * t1 * pitch + 0.5))
            + 5 * np.exp(2j * np.pi * (400 * t1 * pitch + 0.5))
        )
        t2 = np.arange(N, 2 * N, 1) / sample_rate
        v2 = (
            7 * np.exp(2j * np.pi * (100 * t2 * pitch + 0.5))
            + 2 * np.exp(2j * np.pi * (250 * t2 * pitch + 0.5))
            + 2 * np.exp(2j * np.pi * (300 * t2 * pitch + 0.5))
        )
        expected = np.real(np.concatenate([v1, v2]))

        actual = self.note.listen(sample_rate, pitch, 1., 1.)

        assert abs(expected - actual).sum() < expected.shape[0] * (10 ** - 10)


    def test_duration(self):
        # we're going to set the phases so we know what to expect
        self.note.phases = defaultdict(lambda: 0.5)
        sample_rate = 100
        duration = 0.9

        N = sample_rate * self.note.sample_time * duration
        t1 = np.arange(0, N, 1) / sample_rate
        v1 = (
            10 * np.exp(2j * np.pi * (100 * t1 + 0.5)) 
            + 2 * np.exp(2j * np.pi * (250 * t1 + 0.5))
            + 5 * np.exp(2j * np.pi * (400 * t1 + 0.5))
        )
        t2 = np.arange(N, 2 * N, 1) / sample_rate
        v2 = (
            7 * np.exp(2j * np.pi * (100 * t2 + 0.5))
            + 2 * np.exp(2j * np.pi * (250 * t2 + 0.5))
            + 2 * np.exp(2j * np.pi * (300 * t2 + 0.5))
        )
        expected = np.real(np.concatenate([v1, v2]))

        actual = self.note.listen(sample_rate, 1., duration, 1.)

        assert abs(expected - actual).sum() < expected.shape[0] * (10 ** - 10)

    def test_amplitude(self):
        # we're going to set the phases so we know what to expect
        self.note.phases = defaultdict(lambda: 0.5)
        sample_rate = 100
        amplitude = 1.5

        N = sample_rate * self.note.sample_time
        t1 = np.arange(0, N, 1) / sample_rate
        v1 = (
            amplitude * 10 * np.exp(2j * np.pi * (100 * t1 + 0.5)) 
            + amplitude * 2 * np.exp(2j * np.pi * (250 * t1 + 0.5))
            + amplitude * 5 * np.exp(2j * np.pi * (400 * t1 + 0.5))
        )
        t2 = np.arange(N, 2 * N, 1) / sample_rate
        v2 = (
            amplitude * 7 * np.exp(2j * np.pi * (100 * t2 + 0.5))
            + amplitude * 2 * np.exp(2j * np.pi * (250 * t2 + 0.5))
            + amplitude * 2 * np.exp(2j * np.pi * (300 * t2 + 0.5))
        )
        expected = np.real(np.concatenate([v1, v2]))

        actual = self.note.listen(sample_rate, 1., 1., amplitude)

        assert abs(expected - actual).sum() < expected.shape[0] * (10 ** - 10)
