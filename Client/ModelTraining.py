import GameStateViewer
import copy
import cProfile
import pstats
import timeit
import time
import os.path
import socket
from CatanGame import *
from AgentRandom import *
from AgentMCTS import AgentMCTS
from AgentUCT import AgentUCT
from AgentRAVE import AgentRAVE
from AgentAlphabeta import AgentAlphabeta
from AgentPolicy import AgentPolicy, PolicyNetwork
from AgentRandom2 import AgentRandom2
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



# Runs game loop for a single game
def RunSingleGame(game: Game) -> (Game, list):

    game = pickle.loads(pickle.dumps(game, -1))

    logProbs = []
    while True:


        currPlayer:Player = game.gameState.players[game.gameState.currPlayer]

        # Returns list of actions chosen by player (here would call agent which would use policy to get actions)
        if currPlayer.seatNumber == 0:
            agentAction, logProb = currPlayer.DoMove(game)
            if logProb != None:
                logProbs.append(logProb)
        else:
            agentAction = currPlayer.DoMove(game)

        agentAction.ApplyAction(game.gameState)

        if game.gameState.currState == "OVER":
            return game, logProbs
    


if __name__ == '__main__':

    NUM_SIMULATIONS = 5000
    STATE_SIZE = 2350
    ACTION_SIZE = 528

    winRewardTracker = []
    vpTracker = []

    network = PolicyNetwork(STATE_SIZE, ACTION_SIZE, 256)

    players = [
        AgentPolicy("P0", 0, network, playerTrading=False),
        AgentRandom2("P1", 1, playerTrading=False),
        AgentRandom2("P2", 2, playerTrading=False),
        AgentRandom2("P3", 3, playerTrading=False)
    ]

    start_time = time.time()
    results = [0, 0, 0, 0]



    for episode in range(0, NUM_SIMULATIONS):
        inGame = CreateGame(players)
        game, logProbs = RunSingleGame(inGame)
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
        agent.network.update_policy(victoryPts, logProbs)

        results[winner] += 1

        if episode % 10 == 0:
            sys.stdout.write("episode: {}, average_reward: {}\n".format(episode, np.round(np.sum(vpTracker[-10:])/10, decimals = 3)))

    end_time = time.time()

    print(f"\n\nResults: {results}")
    print("Time: ", end_time-start_time)

    plt.plot(vpTracker)
    plt.xlabel('Episode')
    plt.plot(winRewardTracker)
    plt.xlabel('Episode')
    plt.show()

