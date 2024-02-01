

# Use model for initial placement phase (up to the first PLAY)
# Then switch to using random moves for the rest of the game
# the reward will be winning the game to start with

import GameStateViewer
import copy
import cProfile
import pstats
import timeit
import time
from CatanGame import *
from Agents.AgentRandom import *
from Agents.AgentPolicy import AgentPolicy
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentSetup import AgentSetup
from joblib import Parallel, delayed
import multiprocessing
from CatanUtilsPy import listm
import random
import pickle
from ModelState import getInputState
import numpy as np
import matplotlib.pyplot as plt
from CatanBoard import numberDotsMapping

from CatanSimulator import CreateGame

from PPO import PPO

# Takes in list of production for each dice number and returns weighted sum
def getProductionReward(productionDict: dict) -> int:
    reward = 0
    for diceNumber, list in productionDict.items():
        reward += numberDotsMapping[diceNumber] * sum(list)
    return reward

# Runs game loop for a single game
def RunSingleGame(game: Game) -> (Game, list):

    game = pickle.loads(pickle.dumps(game, -1))

    while True:

        currPlayer:Player = game.gameState.players[game.gameState.currPlayer]
        # Returns list of actions chosen by player (here would call agent which would use policy to get actions)
        agentAction = currPlayer.DoMove(game)
        agentAction.ApplyAction(game.gameState)

        if game.gameState.currState == "PLAY":
            return game
        

################ PPO hyperparameters ################


K_epochs = 5#40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.003#0.0003       # learning rate for actor network
lr_critic = 0.01#0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

UPDATE_FREQ = 4

STATE_SIZE = 54
ACTION_SIZE = 54

TRAIN = True
NUM_SIMULATIONS = 1000

if __name__ == '__main__':

    winList = []
    winList100 = []
    vpList = []
    vpList100 = []
    productionRewardList = []
    productionRewardList100 = []

    ppo = PPO(STATE_SIZE, ACTION_SIZE, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    players = [
        AgentSetup("P0", 0, ppo),
        AgentRandom2("P1", 1),
        AgentRandom2("P2", 2),
        AgentRandom2("P3", 3)
    ]

    start_time = time.time()
    results = [0, 0, 0, 0]


    for episode in range(0, NUM_SIMULATIONS):
        inGame = CreateGame(players)
        game = RunSingleGame(inGame)

        # winner = game.gameState.winner

        agent:AgentPolicy = game.gameState.players[0]

        # winReward = 0
        # if winner == 0:
        #     winReward = 3
        # else:
        #     winReward = -1
        # victoryPts = agent.victoryPoints
        # vpList.append(victoryPts)
        # winList.append(winReward)

        production = getProductionReward(agent.diceProduction)
        productionRewardList.append(production-18)

        agent.network.buffer.rewards[-2:] = 2*[production-18]
        # agent.network.buffer.is_terminals[-1] = True

        if TRAIN:
            if episode % UPDATE_FREQ == 0:
                agent.network.update(disableDiscountReward=True)

        # results[winner] += 1

        if episode % 100 == 0:
            # winList100.append(np.round(np.sum(winList[-100:])/100, decimals = 3))
            # vpList100.append(np.round(np.sum(vpList[-100:])/100, decimals = 3))
            productionRewardList100.append(np.round(np.sum(productionRewardList[-100:])/100, decimals = 3))
            # print("episode: {}, average_vp: {}, average_wins: {}\n".format(episode, vpList100[-1], winList100[-1]))
            print("episode: {}, productionReward: {}\n".format(episode, productionRewardList100[-1]))


    end_time = time.time()

    # print(f"\n\nResults(Training:{TRAIN}): {results}")
    print("Time: ", end_time-start_time)


    plt.plot(productionRewardList100)
    plt.xlabel('Episode (100)')
    plt.ylabel('Avg VP')
    plt.show()
