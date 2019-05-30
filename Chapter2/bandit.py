import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import e, pow

np.random.seed(5)

class Environment():
    
    def __init__(self, armCount = 10, mean = 0.0, stddev = 1.0, nonStationarity = 0.0):
        
        self.armCount = armCount
        self.mean = mean
        self.stddev = stddev
        self.arms = np.arange(armCount)
        
        if nonStationarity == 0.0:
            self.qStar = np.random.normal(mean, stddev, armCount)
            vMax = max(self.qStar)
            self.optimalChoice = np.random.choice(np.flatnonzero(self.qStar == vMax))
        else:
            self.qStar = np.zeros(armCount)
            self.qStar.fill(1/armCount) # all reward are equal initialy
            self.optimalChoice = np.random.choice(np.flatnonzero(self.qStar == (1/armCount)))
        
        self.randomWalk = nonStationarity # stddiv for random walk
            
    def reward(self, chosen):
        if self.randomWalk != 0.0:
            self.qStar += np.random.normal(0.0, scale = self.randomWalk, size = self.armCount)
            vMax = max(self.qStar)
            self.optimalChoice = np.random.choice(np.flatnonzero(self.qStar == vMax))
        return np.random.normal(self.qStar[chosen], scale = 1.0)
    
    def refresh(self):
        if self.randomWalk == 0.0:
            self.qStar = np.random.normal(self.mean, self.stddev, self.armCount)
            vMax = max(self.qStar)
            self.optimalChoice = np.random.choice(np.flatnonzero(self.qStar == vMax))
        else:
            self.qStar.fill(1/self.armCount) 
            self.optimalChoice = np.random.choice(np.flatnonzero(self.qStar == (1/self.armCount)))

class Agent():

    def __init__(self, armCount, epsilon = 0.0, decay = 0.0,
                     alpha = 0.0, init = 0, c = 0, temp = 0.0, gradient = False,
                     baselineGradient = False, softmax = False):

        self.values = np.zeros(armCount) # value estimation Q(action)
        self.init = init # initial value (Optimistic)
        if init != 0 : 
            self.values.fill(init) # initial value (Optimistic)
        self.pullCount = np.zeros(armCount) # number of selection for each action
        self.choice = np.arange(armCount) # choice array for picking action [0, armCount-1]
        self.armCount = armCount # number of action
        self.epsilon = epsilon # (EpsilonGreedy, Const step-size)
        self.decay = decay # decay rate (Decay)
        self.time = 0 # keep track of time (Decay, UCB, Gradient)
        self.c = c # uncertainty factor (UCB)
        self.alpha = alpha # const step-size (Const step-size, Optimistic)
        self.baselineGradient = baselineGradient # baseline (Gradient)
        self.rAvg = 0 # average of all recieved rewards (Gradient)
        self.gradient = gradient # (Gradient)
        self.preference = np.zeros(armCount) # H_t(a) (Gradient)
        self.prob = np.zeros(armCount) #  π_t(a) (Gradient)
        self.prob.fill(1/armCount) # initially H_t(a) = 0 cause each prob to be equal (Gradient, Softmax)
        self.softmax = softmax # (Softmax)
        self.temp = temp # tau (Softmax)
        self.valueExp = np.ones(armCount) # save e ^ value/temp (Softmax)
        self.sumExp = 1 * self.armCount # sum of all e ^ value/temp 
                                            # (in beggining all of them are 1) (Softmax)

    def __repr__(self):
        if self.gradient:
            return "Gradient Alpha= " + str(self.alpha) + (" with baseline" if self.baselineGradient else " without baseline")
        if self.c != 0:
            return "UCB: " + str(self.c)
        elif self.init != 0 :
             return "Optimistic Q0=" + str(self.init) + (" Greedy" if self.epsilon == 0 else " Epsilon Greedy= " + str(self.epsilon))
        elif self.decay != 0.0 :
             return "Epsilon Decay= " + str(self.decay) + ("" if self.alpha == 0 else " Constant step size= " + str(self.alpha))
        elif self.epsilon != 0.0 :
             return "Epsilon Greedy= " + str(self.epsilon) + ("" if self.alpha == 0 else " Constant step size= " + str(self.alpha))
        else : 
            return "Greedy"

    def selectArm(self):

        self.time += 1

        if self.decay != 0.0: self.epsilon = 1/(1+self.time*self.decay)
        
        if self.gradient or self.softmax:
            return np.random.choice(self.choice, p=self.prob)
        elif self.c != 0:
            ubc = self.values + self.c * np.sqrt(np.log(self.time)/self.pullCount)
            return np.argmax(ubc)
        else:
            if np.random.rand() > self.epsilon: # Exploit
                vMax = max(self.values)
                return np.random.choice(np.flatnonzero(self.values == vMax))
            else: # Explore
                return np.random.choice(self.choice)

    def update(self, i, reward):

        if self.gradient:
            if self.baselineGradient : self.rAvg += (1/self.time) * (reward - self.rAvg)
            for j, pre in enumerate(self.preference):
                if i == j : 
                    self.preference[j] += (self.alpha) * (reward - self.rAvg) * (1 - self.prob[j])
                else : 
                    self.preference[j] -= (self.alpha) * (reward - self.rAvg) * (self.prob[j])
            self.prob = np.exp(self.preference)
            self.prob /= np.sum(self.prob)
        elif self.softmax:
            self.pullCount[i] += 1
            count = self.pullCount[i]
            value = self.values[i]
            # updating value estimation and probabilties
            self.sumExp -= self.valueExp[i]
            self.values[i] = value + (1/count) * (reward - value) #* (1 - prob)
            m = self.values[i] / self.temp
            self.valueExp[i] = pow(e,m)
            self.sumExp += self.valueExp[i]
            self.prob = self.valueExp / self.sumExp
        else:
            self.pullCount[i] += 1
            count = self.pullCount[i]
            value = self.values[i]

            # newEstimate = oldEstimate + step_size * (target - oldEstimate)
            if self.alpha == 0 : self.values[i] = value + (1/count) * (reward - value)
            else : self.values[i] = value + (self.alpha) * (reward - value)

    def refresh(self):

        self.values.fill(0)
        self.pullCount.fill(0)
        if self.init != 0 :  self.values.fill(self.init)
        self.time = 0
        if self.gradient or self.softmax:
            self.preference.fill(0)
            self.prob.fill(1/self.armCount)
            self.rAvg = 0
        if self.softmax:
            self.valueExp.fill(1.0)
            self.sumExp = 1 * self.armCount

class Sampling():
    
    def __init__(self, env, agents, play = 1000, run = 2000):
        self.env = env
        self.agents = agents
        self.play = play
        self.run = run
        self.rewardData = np.zeros((len(agents), play, run))
        self.optimalityData = np.zeros((len(agents), play, run))
        
    def start(self):
        
        for r in tqdm(range(self.run)):
            for p in range(self.play):
                for a in range(len(self.agents)):
                    choosen = self.agents[a].selectArm()
                    reward = self.env.reward(choosen)
                    self.agents[a].update(choosen, reward)
                    self.rewardData[a][p][r] += reward
                    if self.env.optimalChoice == choosen:
                        self.optimalityData[a][p][r] += 1
        
            self.env.refresh()
            for a in range(len(agents)):
                self.agents[a].refresh()

def plotResults(sample):
    rewardMean = np.mean(sample.rewardData, axis=2)
    optimalityMean = np.mean(sample.optimalityData, axis=2)
    time = np.array(list(range(sample.play)))

    for a in range(len(sample.agents)):
        plt.subplot(2, 1, 1)
        plt.plot(time, optimalityMean[a], '-o', markersize=0.7, lw=1.5, 
                    label=str(sample.agents[a]))
        plt.ylabel('% Optimal Action')

        plt.subplot(2, 1, 2)
        plt.plot(time, rewardMean[a], '-o', markersize=0.7, lw=1.5, 
                    label=str(sample.agents[a]))
        plt.ylabel('Average Reward')
        plt.xlabel('play')

    plt.legend(loc='best', fancybox=True)
    plt.show()

def plotSummery(sample, parameters, labels, cFrom = 0):
    rewardMean = np.mean(sample.rewardData[:,cFrom:,:], axis=2)
    agentsChoice = np.arange(len(labels))
    i=0
    for a, parameter in zip(agentsChoice, parameters):
            l = len(parameter)
            plt.plot(parameter, np.mean(rewardMean[i:i+l], axis=1)
                    , markersize=0.7, lw=1.5, label=labels[a])
            i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend(loc='best', fancybox=True)
    plt.show()

# Stationary           -->   Environment()
# Non-Stationary 0.01  -->   Environment(nonStationarity=0.01)

env = Environment(nonStationarity=0.01)

# Greedy               -->   Agent(env.armCount)
# Epsilon Greedy 0.1   -->   Agent(env.armCount, epsilon=0.1)
# Decay Greedy 0.1     -->   Agent(env.armCount, decay=0.1)
# Const step size 0.1  -->   Agent(env.armCount, epsilon=0.1, alpha=0.1) or Agent(env.armCount, decay=0.1, alpha=0.1)
# Optimistic 0.1 5     -->   Agent(env.armCount, epsilon=0.1, alpha=0.1, init=5)
# UCB 2                -->   Agent(env.armCount, c=2)
# Gradient true 0.1    -->   Agent(env.armCount, alpha=0.1, gradient=True, baselineGradient=True)
# Softmax 0.25         -->   Agent(env.armCount, softmax=True, temp=0.25)

agents = [Agent(env.armCount, epsilon=0.1),
          Agent(env.armCount, alpha=0.1, epsilon=0.1)]

sam = Sampling(env, agents)
sam.start()
plotResults(sam)

