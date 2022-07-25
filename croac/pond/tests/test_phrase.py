import unittest
import numpy as np

from ..phrase import (
    Phrase
)

class MockNote(object):
    @staticmethod
    def listen(*args):
        return np.array(args)

class TestPhraseListen(unittest.TestCase):
    def test_initial_pause(self):
        phrase = Phrase(
            [MockNote()],
            [1],
            [{}],
            initial_pause=1.2
        )
        expected = np.array([
            0, 0, 0, 0, 3, 1, 1, 1, 0, 0, 0
        ])
        actual = phrase.listen(3, 1., 1., 1.)

        assert (expected == actual).all()
    
    def test_pauses(self):
        phrase = Phrase(
            [MockNote(), MockNote()],
            [1, 2.3],
            [{}, {}],
            initial_pause=1.2
        )
        expected = np.array([
            0, 0, 0, 0, 3, 1, 1, 1, 0, 0, 0, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
        ])
        actual = phrase.listen(3, 1., 1., 1.)

        assert (expected == actual).all()

    def test_shapes(self):
        phrase = Phrase(
            [MockNote(), MockNote()],
            [1, 2.3],
            [{'amplitude': 1.2}, {'duration': 0.1, 'pitch': 1.1},],
            initial_pause=1.2
        )
        expected = np.array([
            0, 0, 0, 0, 3, 2, 3, 4.8, 0, 0, 0, 3, 2.2, 0.3, 4, 0, 0, 0, 0, 0, 0, 0
        ])
        actual = phrase.listen(3, 2., 3., 4.)

        print(expected - actual)

        assert abs(expected - actual).sum() < expected.shape[0] * (10 ** -10)
