import numpy as np

SPEED_OF_SOUND = 343 # m/s

class Pond(object):
    def __init__(self, frogs, positions, antennas):
        self.frogs = frogs
        self.positions = positions
        self.antennas = antennas

    def listen_to_antenna(self, antenna, sample_rate=44100):
        assert len(self.frogs) == len(self.positions)
        distances = [
            np.linalg.norm(antenna - position)
            for position in self.positions
        ]
        sample_delays = [
            round(distance / SPEED_OF_SOUND * sample_rate)
            for distance in distances
        ]
        songs = [
            np.concatenate([np.zeros(delay), frog.listen(sample_rate) / (distance ** 2)])
            for distance, delay, frog in zip(distances, sample_delays, self.frogs)
        ]
        max_song_length = max(song.shape[0] for song in songs)
        songs = np.array([
            np.concatenate([song, np.zeros(max_song_length - song.shape[0])])
            for song in songs
        ])
        chorus = songs.sum(axis=0)
        return chorus

    def listen(self, sample_rate=44100):
        return [
            self.listen_to_antenna(antenna, sample_rate)
            for antenna in self.antennas
        ]
