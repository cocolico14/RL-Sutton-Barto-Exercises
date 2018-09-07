import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import Stats
import sys, os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class EpsilonGreedy():

    def __init__(self, armCount, epsilon=0.1, alpha = None):
        self.values = np.zeros(armCount)
        self.pullCount = np.zeros(armCount)
        self.armCount = armCount
        self.epsilon = epsilon
        self.alpha = alpha

    def __repr__(self):
        return "Epsilon: " + str(self.epsilon)

    def selectArm(self):
        if np.random.rand() > self.epsilon: # Exploit
            return np.argmax(self.values)
        else: # Explore
            return int(np.floor(np.random.uniform(high=self.armCount)))

    def update(self, i, reward):
        self.pullCount[i] += 1
        count = self.pullCount[i]
        value = self.values[i]
        # newEstimate = oldEstimate + step_size * (target - oldEstimate)
        self.values[i] = value + (1/count) * (reward - value)

    def refresh(self):
        self.values.fill(0)
        self.pullCount.fill(0)

class Softmax():

    def __init__(self, armCount, temp=0.1):
        self.values = np.zeros(armCount)
        self.pullCount = np.zeros(armCount)
        self.armCount = armCount
        self.choice = np.array(list(range(armCount)))
        self.temp = temp

    def __repr__(self):
        return "Softmax: " + str(self.temp)

    def selectArm(self):
        probability = np.exp(self.values/self.temp)
        sumExp = np.sum(probability)
        probability /= sumExp
        return np.random.choice(self.choice, p=probability)

    def update(self, i, reward):
        self.pullCount[i] += 1
        count = self.pullCount[i]
        value = self.values[i]
        prob = np.exp(value/self.temp)/np.sum(np.exp(self.values/self.temp))
        self.values[i] = value + (1/count) * (reward - value) * (1 - prob)
        

    def refresh(self):
        self.values.fill(0)
        self.pullCount.fill(0)

class Arm():

    def __init__(self):
        self.prob = np.random.rand()

    def pull(self):
        return 1 if np.random.rand()<self.prob else 0

    def getOptimality(self):
        return self.prob

    def __iter__(self):
        while True:
            yield Arm()

    def __repr__(self):
        rep = "{0:.2f}%".format(self.prob*100)
        return rep



if __name__ == '__main__':

    armCount = 10
    epochs = 1000
    play = 2000

    blockPrint()
    param = [{'_algo': Softmax(armCount, 1), '_epochs': epochs,
             '_optAct': None, '_reward': None, 
             '_optActMean': None, '_rewardMean': None, '_color': 'blue'}
             
             ]

    cases = [Stats(**(param[i])) for i in range(len(param))]
    enablePrint()

    for p in tqdm(range(play)):

        np.random.seed(p)

        armGen = iter(Arm())
        arms = [next(armGen) for i in range(armCount)]
        cumulativeReward = 0
        optimalChoice = np.argmax([arm.getOptimality() for arm in arms])

        for i in range(armCount):
            reward = arms[i].pull()
            for c in range(len(cases)):
                cases[c].algorithm.update(i, reward)

        for e in range(epochs):
            for c in range(len(cases)):
                choosen = cases[c].algorithm.selectArm()
                reward = arms[choosen].pull()
                cases[c].algorithm.update(choosen, reward)
                cases[c].reward[e] += reward
                if optimalChoice == choosen:
                    cases[c].optimalAction[e] += 1
        
        for c in range(len(cases)):
            cases[c].optimalActionMean += cases[c].optimalAction
            cases[c].rewardMean += cases[c].reward
            cases[c].optimalAction.fill(0)
            cases[c].reward.fill(0)
            cases[c].algorithm.refresh()

        # pullCount = [int(c) for c in bandit.pullCount]
        # for (k, v) in zip(arms, pullCount):
        #     print(str(k) + " = " + str(v) + " pulls")
        # print("\nTotal Reward: " + str(cumulativeReward))

    
    for c in range(len(cases)):
        cases[c].rewardMean /= play
        cases[c].optimalActionMean /= play
        time = np.array(list(range(epochs)))
        plt.subplot(2, 1, 1)
        plt.plot(time, cases[c].optimalActionMean, '-o'
                , color=cases[c].color, markersize=0.7, lw=1.5, label=str(cases[c].algorithm))
        plt.ylabel('% Optimal Action')

        plt.subplot(2, 1, 2)
        plt.plot(time, cases[c].rewardMean, '-o'
                , color=cases[c].color, markersize=0.7, lw=1.5, label=str(cases[c].algorithm))
        plt.ylabel('Average Reward')
        plt.xlabel('Epochs')

    plt.legend(loc='best')
    plt.axis([0, epochs, 0, 1])
    plt.show()
        



