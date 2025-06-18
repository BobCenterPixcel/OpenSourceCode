import random


class VolumePerturbAugmentor(object):
    def __init__(self, min_gain_dBFS=-15, max_gain_dBFS=15, prob=0.5):
        self.prob = prob
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS

    def __call__(self, wav):
        if random.random() > self.prob: return wav
        gain = random.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        wav *= 10.**(gain / 20.)
        return wav
