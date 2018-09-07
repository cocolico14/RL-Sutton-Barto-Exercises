# pylint: disable-msg=E1101
import numpy as np

class Stats():

    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            print(attr, value)
            setattr(self, attr, value)

    @property
    def epochs(self):
        if self._epochs is None:
            self._epochs = 2000
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @property
    def algorithm(self):
        return self._algo

    @property
    def optimalAction(self):
        if self._optAct is None:
            self._optAct = np.zeros(self._epochs)
        return self._optAct

    @optimalAction.setter
    def optimalAction(self, value):
        self._optAct = value

    @property
    def reward(self):
        if self._reward is None:
            self._reward = np.zeros(self._epochs)
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value

    @property
    def optimalActionMean(self):
        if self._optActMean is None:
            self._optActMean = np.zeros(self._epochs)
        return self._optActMean

    @optimalActionMean.setter
    def optimalActionMean(self, value):
        self._optActMean = value

    @property
    def rewardMean(self):
        if self._rewardMean is None:
            self._rewardMean = np.zeros(self._epochs)
        return self._rewardMean

    @rewardMean.setter
    def rewardMean(self, value):
        self._rewardMean = value

    @property
    def color(self):
        return self._color