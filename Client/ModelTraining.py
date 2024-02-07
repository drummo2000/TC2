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



NUM_SIMULATIONS = 1
STATE_SIZE = 2350
ACTION_SIZE = 486


network = VPGNetwork(STATE_SIZE, ACTION_SIZE, 256)


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

##############################################################

env = CatanSetupEnv() #CatanEnv(ACTION_SIZE, STATE_SIZE)

start_time = time.time()

for i in range(NUM_SIMULATIONS):
    done = False
    state, actionMask = env.reset(players, custom_board)

    while done != True:
        actionIndex = network.get_action(state, actionMask)
        state, actionMask, reward, done = env.step(actionIndex)

end_time = time.time()

print(f"Time: {end_time-start_time}")