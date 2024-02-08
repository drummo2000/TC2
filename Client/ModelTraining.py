import GameStateViewer
import copy
import cProfile
import pstats
import timeit
import time
import os.path
import socket
from CatanGame import *
from Agents.AgentRandom2 import AgentRandom2
from joblib import Parallel, delayed
import multiprocessing
import CSVGenerator
from CatanUtilsPy import listm
import random
import pickle
from ModelState import getInputState
import numpy as np
import matplotlib.pyplot as plt

from CatanSimulator import CreateGame
from VPG import VPGNetwork, AgentVPG
from CatanEnv import CatanEnv, CatanSetupEnv
from PPO import PPO


#               GAME SETTINGS
##############################################################


NUM_SIMULATIONS = 100
STATE_SIZE = 2350
ACTION_SIZE = 486

players = [
    AgentRandom2("P0", 0),
    AgentRandom2("P1", 1),
    AgentRandom2("P2", 2),
    AgentRandom2("P3", 3)
]

custom_board = "1014|TestGame,7,6,20,6,6,2,3,5,34,53,4,1,3,1," \
                     "6,6,4,5,0,4,2,8,49,5,2,4,3,6,6,1,4,3,67,9,6," \
                     "10,6,-1,-1,-1,-1,-1,7,0,6,-1,-1,9,4,2,7,-1,-1," \
                     "6,8,-1,1,5,-1,-1,5,1,2,3,-1,-1,3,4,8,-1,-1,-1,-1,-1,85"

env = CatanSetupEnv() #CatanEnv(ACTION_SIZE, STATE_SIZE)

##############################################################

#           NETWORK SETTINGS

network = PPO(54, 54, K_epochs=1, lr_actor=0.03, lr_critic=0.01)

TRAIN_FREQ = 10


##############################################################
rewardList = []
rewardList100 = []

start_time = time.time()

for episode in range(NUM_SIMULATIONS):
    done = False
    state, info = env.reset(players, custom_board)
    actionMask = info["ActionMask"]

    while done != True:
        actionIndex = network.select_action(state, actionMask)
        state, reward, done, _, info = env.step(actionIndex)
        actionMask = info["ActionMask"]
        
        rewardList.append(reward)

        network.buffer.rewards.append(reward)
        network.buffer.is_terminals.append(done)

    if episode % TRAIN_FREQ == 0:
        network.update()

    if episode % 100 == 0:
            rewardList100.append(np.round(np.sum(rewardList[-100:])/100, decimals = 3))
            print("episode: {}, Reward: {}\n".format(episode, rewardList100[-1]))

end_time = time.time()

print(f"Time: {end_time-start_time}")

plt.plot(rewardList100)
plt.xlabel('Episode (100)')
plt.ylabel('Avg VP')
plt.show()