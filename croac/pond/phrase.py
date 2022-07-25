import numpy as np

class Phrase(object):
    def __init__(self, notes, pauses, shapes, initial_pause=0.):
        self.notes = notes
        self.pauses = pauses
        self.shapes = shapes
        self.initial_pause = initial_pause
        
    def listen(self, sample_rate=44100, pitch=1., duration=1., amplitude=1.):
        assert len(self.notes) == len(self.pauses)
        assert len(self.notes) == len(self.shapes)
        sound = np.zeros(round(self.initial_pause * sample_rate))
        for note, pause, shape in zip(self.notes, self.pauses, self.shapes):
            sound = np.concatenate([
                sound, 
                note.listen(
                    sample_rate,
                    shape.get('pitch', 1) * pitch,
                    shape.get('duration', 1) * duration,
                    shape.get('amplitude', 1) * amplitude
                ),
                np.zeros(round(pause * sample_rate))
            ])
        return sound
