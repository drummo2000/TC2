import gym
import numpy as np
from gym import spaces
from CatanSimulator import CreateGame
import pickle
import GameStateViewer
import copy
import cProfile
import pstats
import timeit
import time
import os.path
import socket
from CatanGame import *
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
from ModelState import getInputState
from ActionMask import getActionMask, getSetupActionMask
from CatanPlayer import Player

# Takes in list of production for each dice number and returns weighted sum
def getProductionReward(productionDict: dict) -> int:
    reward = 0
    for diceNumber, list in productionDict.items():
        reward += numberDotsMapping[diceNumber] * sum(list)
    return reward


class CatanEnv(gym.Env):
    def __init__(self, action_size: int, state_size: int):
        super(CatanEnv, self).__init__()

        self.action_space = spaces.Discrete(action_size)  
        self.observation_space = spaces.Discrete(state_size) 
        self.game: Game = None
        self.indexActionDict = None
        self.players = []
        self.agent = None

    # Need to get to my players turn and return state and action mask
    def reset(self, players, customBoard=None):
        # Setup game
        inGame = CreateGame(players, customBoard)
        self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through until agents turn
        currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        actionMask, self.indexActionDict = getActionMask(possibleActions)
        return getInputState(self.game.gameState), actionMask


    # Takes in index of agents action and returns state, action_mask, reward, done
    def step(self, action):
        done = False

        # Apply action chosen by agent
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if self.game.gameState.currState == "OVER":
            return None, None, 0, True
        # if game is not over cycle through actions until its agents turn again
        else:
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]
            while currPlayer.seatNumber != 0:
                agentAction = currPlayer.DoMove(self.game)
                agentAction.ApplyAction(self.game.gameState)
                if self.game.gameState.currState == "OVER":
                    return None, None, 0, True
                currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]
        
        # Get necessary info to return
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        actionMask, self.indexActionDict = getActionMask(possibleActions)

        # observation, action_mask, done, reward
        return getInputState(self.game.gameState), actionMask, 0, done


class CatanSetupEnv(gym.Env):
    def __init__(self):
        super(CatanSetupEnv, self).__init__()

        self.action_space = spaces.Discrete(54)  
        self.observation_space = spaces.Discrete(54) 
        self.game: Game = None
        self.indexActionDict = None
        self.players = []
        self.agent = None
        self.lastReward = 0

    # Need to get to my players turn and return state and action mask
    def reset(self, players, customBoard=None):
        # Setup game
        inGame = CreateGame(players, customBoard)
        self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]
        self.lastReward = 0

        # Cycle through until agents turn
        currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        actionMask, self.indexActionDict = getSetupActionMask(possibleActions)
        return getInputState(self.game.gameState), actionMask


    # Takes in index of agents action and returns state, action_mask, reward, done
    def step(self, action):
        done = False

        # Apply action chosen by agent
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        # Get reward for action
        reward = getProductionReward(self.agent.diceProduction) - self.lastReward
        self.lastReward = reward

        adjReward = reward -5

        if self.game.gameState.currState == "PLAY":
            return None, None, adjReward, True
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0 or self.game.gameState.currState == "START1B" or self.game.gameState.currState == "START2B":
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            if self.game.gameState.currState == "PLAY":
                return None, None, adjReward, True
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]
        
        # Get necessary info to return
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        actionMask, self.indexActionDict = getSetupActionMask(possibleActions)

        # observation, action_mask, done, reward
        return getInputState(self.game.gameState), actionMask, adjReward, done