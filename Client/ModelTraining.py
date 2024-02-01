import GameStateViewer
import copy
import cProfile
import pstats
import timeit
import time
import os.path
import socket
from CatanGame import *
from Agents.AgentRandom import *
from Client.Agents.AgentMCTS import AgentMCTS
from Agents.AgentUCT import AgentUCT
from Agents.AgentRAVE import AgentRAVE
from Agents.AgentAlphabeta import AgentAlphabeta
from Agents.AgentPolicy import AgentPolicy
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

from PPO import PPO



# Runs game loop for a single game
def RunSingleGame(game: Game) -> (Game, list):

    game = pickle.loads(pickle.dumps(game, -1))

    while True:


        currPlayer:Player = game.gameState.players[game.gameState.currPlayer]

        # Returns list of actions chosen by player (here would call agent which would use policy to get actions)
        agentAction = currPlayer.DoMove(game)

        agentAction.ApplyAction(game.gameState)

        if game.gameState.currState == "OVER":
            return game
        

################ PPO hyperparameters ################


K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

UPDATE_FREQ = 10

STATE_SIZE = 2351
ACTION_SIZE = 528

TRAIN = False
NUM_SIMULATIONS = 100

if __name__ == '__main__':

    winRewardTracker = []
    vpTracker = []

    ppo = PPO(STATE_SIZE, ACTION_SIZE, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    players = [
        AgentPolicy("P0", 0, ppo),
        AgentRandom2("P1", 1),
        AgentRandom2("P2", 2),
        AgentRandom2("P3", 3)
    ]

    start_time = time.time()
    results = [0, 0, 0, 0]

    avgVPList = []

    for episode in range(0, NUM_SIMULATIONS):
        inGame = CreateGame(players)
        game = RunSingleGame(inGame)
        winner = game.gameState.winner

        agent:AgentPolicy = game.gameState.players[0]

        winReward = 0
        if winner == 0:
            winReward = 1
        else:
            winReward = -1
        victoryPts = agent.victoryPoints
        vpTracker.append(victoryPts)
        winRewardTracker.append(winReward)

        agent.network.buffer.rewards[-1] = victoryPts
        agent.network.buffer.is_terminals[-1] = True

        if TRAIN:
            if episode % UPDATE_FREQ == 0:
                agent.network.update()

        results[winner] += 1

        # if episode % 100 == 0:
        #     print("episode: {}, average_reward: {}\n".format(episode, np.round(np.sum(vpTracker[-100:])/100, decimals = 3)))
        #     avgVPList.append(np.round(np.sum(vpTracker[-100:])/100, decimals = 3))
        #     print(f"CurrentResults: {results}")

    end_time = time.time()

    print(f"\n\nResults: {results}")
    print("Time: ", end_time-start_time)

    # plt.plot(avgVPList)
    # plt.xlabel('Episode')
    # plt.show()
