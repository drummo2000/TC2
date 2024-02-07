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
from VPG import VPGNetwork, AgentVPG
from GameStateViewer import SaveGameStateImage

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

        agentAction = currPlayer.DoMove(game)
        agentAction.ApplyAction(game.gameState)

        # Once initial placement is over return
        if game.gameState.currState == "START2A":
            # SaveGameStateImage(game.gameState, "TESTIMAGE.png")
            return game


################ PPO hyperparameters ################


K_epochs = 5#40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

UPDATE_FREQ = 1

STATE_SIZE = 54
ACTION_SIZE = 54

TRAIN = True
NUM_SIMULATIONS = 1

if __name__ == '__main__':

    winList = []
    winList100 = []
    vpList = []
    vpList100 = []
    productionRewardList = []
    productionRewardList100 = []

    ppo = PPO(STATE_SIZE, ACTION_SIZE, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    vpg = VPGNetwork(STATE_SIZE, ACTION_SIZE, 256)


    players = [
        AgentSetup("P0", 0, ppo),
        # AgentVPG("P0", 0, vpg),
        AgentRandom2("P1", 1),
        AgentRandom2("P2", 2),
        AgentRandom2("P3", 3)
    ]

    start_time = time.time()
    results = [0, 0, 0, 0]


    for episode in range(0, NUM_SIMULATIONS):
        custom_board = "1014|TestGame,7,6,20,6,6,2,3,5,34,53,4,1,3,1," \
                     "6,6,4,5,0,4,2,8,49,5,2,4,3,6,6,1,4,3,67,9,6," \
                     "10,6,-1,-1,-1,-1,-1,7,0,6,-1,-1,9,4,2,7,-1,-1," \
                     "6,8,-1,1,5,-1,-1,5,1,2,3,-1,-1,3,4,8,-1,-1,-1,-1,-1,85"
        inGame = CreateGame(players, customBoard=custom_board)
        game = RunSingleGame(inGame)

        # winner = game.gameState.winner

        agent:AgentVPG = game.gameState.players[0]

        # winReward = 0
        # if winner == 0:
        #     winReward = 3
        # else:
        #     winReward = -1
        # victoryPts = agent.victoryPoints
        # vpList.append(victoryPts)
        # winList.append(winReward)

        production = getProductionReward(agent.diceProduction)
        productionRewardList.append(production-8)

        agent.network.buffer.rewards[-2:] = [production-8]
        agent.network.buffer.is_terminals[-1] = True

        if TRAIN:
            if episode % UPDATE_FREQ == 0:
                agent.network.update(disableDiscountReward=True)
                # agent.network.update_policy(production-8)

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
