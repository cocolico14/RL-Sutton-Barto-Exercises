import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import pow, e

class EpsilonGreedy():

    def __init__(self, armCount, epsilon=0.1, alpha = None):
        self.values = np.zeros(armCount)
        self.pullCount = np.zeros(armCount)
        self.choice = np.arange(armCount)
        self.armCount = armCount
        self.epsilon = epsilon
        self.alpha = alpha

    def __repr__(self):
        return "Epsilon: " + str(self.epsilon)

    def selectArm(self):
        if np.random.rand() > self.epsilon: # Exploit
            return np.argmax(self.values)
        else: # Explore
            return np.random.choice(self.choice)

    def update(self, i, reward):
        self.pullCount[i] += 1
        count = self.pullCount[i]
        value = self.values[i]
        # newEstimate = oldEstimate + step_size * (target - oldEstimate)
        if self.alpha is None : self.values[i] = value + (1/count) * (reward - value)
        else: self.values[i] = value + (self.alpha) * (reward - value)

    def refresh(self):
        self.values.fill(0)
        self.pullCount.fill(0)

class Softmax():

    def __init__(self, armCount, temp=0.1):
        self.values = np.zeros(armCount)
        self.pullCount = np.zeros(armCount)
        self.temp = temp
        self.armCount = armCount
        self.valueExp = np.ones(armCount)
        self.sumExp = 1 * self.armCount
        self.prob = self.valueExp / self.armCount 
        self.choice = np.arange(armCount)
        

    def __repr__(self):
        return "Softmax: " + str(self.temp)

    def selectArm(self):
        return np.random.choice(self.choice, p=self.prob)

    def update(self, i, reward):
        self.pullCount[i] += 1
        count = self.pullCount[i]
        value = self.values[i]
        self.sumExp -= self.valueExp[i]
        self.values[i] = value + (1/count) * (reward - value) #* (1 - prob)
        m = self.values[i] / self.temp
        self.valueExp[i] = pow(e,m)
        self.sumExp += self.valueExp[i]
        self.prob = self.valueExp / self.sumExp
        

    def refresh(self):
        self.values.fill(0)
        self.pullCount.fill(0)
        self.valueExp.fill(1.0)
        self.sumExp = 1 * self.armCount
        self.prob.fill(1/self.armCount)


if __name__ == '__main__':

    armCount = 10
    play = 1000
    run = 2000

    algos = [EpsilonGreedy(armCount, 0.04, 0.1)]
    colors = ['blue', 'red', 'green', 'magenta', 'cyan']
    rewardData = np.zeros((len(algos), play, run))
    optimalityData = np.zeros((len(algos), play, run))

    arms = np.zeros((run, armCount))

    for r in range(run):
        np.random.seed(r)
        arms[r] = np.random.randn(armCount)

    for r in tqdm(range(run)):

        optimalChoice = np.argmax(arms[r])

        for p in range(play):
            for c in range(len(algos)):
                choosen = algos[c].selectArm()
                reward = np.random.rand()+arms[r][choosen]
                algos[c].update(choosen, reward)
                rewardData[c][p][r] += reward
                if optimalChoice == choosen:
                    optimalityData[c][p][r] += 1
        
        for c in range(len(algos)):
            # pullCount = [int(c) for c in algos[c].pullCount]
            # for (k, v) in zip(arms, pullCount):
            #     print(str(k) + "% = " + str(v) + " pulls")
            algos[c].refresh()

    rewardMean = np.mean(rewardData, axis=2)
    optimalityMean = np.mean(optimalityData, axis=2)
    time = np.array(list(range(play)))

    for c in range(len(algos)):
        plt.subplot(2, 1, 1)
        plt.plot(time, optimalityMean[c], '-o'
                , color=colors[c], markersize=0.7, lw=1.5, label=str(algos[c]))
        plt.ylabel('% Optimal Action')

        plt.subplot(2, 1, 2)
        plt.plot(time, rewardMean[c], '-o'
                , color=colors[c], markersize=0.7, lw=1.5, label=str(algos[c]))
        plt.ylabel('Average Reward')
        plt.xlabel('play')

    plt.legend(loc='best')
    plt.show()
        



