import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Game.CatanPlayer import Player
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentModel import AgentModel
from CatanData.GameStateViewer import SaveGameStateImage
from DeepLearning.GetActionMask import getActionMask, getActionMaskTrading
from DeepLearning.PPO import MaskablePPO
from CatanData.GameStateViewer import SaveGameStateImage, DisplayImage
import time
from collections import deque
from DeepLearning.globals import GAME_RESULTS
from DeepLearning.Environments.CatanEnv import CatanBaseEnv
from DeepLearning.GetActionMask import getActionMask
from DeepLearning.Thesis.Observations.get_observation import getObservation, lowerBound, upperBound


class SetupRandom(CatanBaseEnv):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """
    def __init__(self, customBoard=None, players=None, trading=False):
        super(SetupRandom, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 10
        self.loseRewardAmount = -10
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
            # Trading Rewards
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = True
        self.denseRewardMultiplier = 1

        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMask
        self.getObservation = getObservation

    
    def reset(self, seed=None):
        self.numTurns = 0
        return super(SetupRandom, self).reset()

    def endCondition(self) -> bool:
        if self.game.gameState.currState == "PLAY":
            return True
        else:
            return False


    def step(self, action):
        """
        Accepts action index as argument, applies action, cycles through to players next turn, 
        gets observation and action mask for turn
        """
        truncated = False
        done = False

        reward = 0

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1

        # Check if game Over
        if self.endCondition():
            return self.endGame(reward)
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while True:
            # Only use model when right turn and more than 1 possible action
            if currPlayer.seatNumber == 0:
                possibleActions = self.agent.GetPossibleActions(self.game.gameState)
                if len(possibleActions) > 1:
                    break
                elif possibleActions[0].type == "EndTurn":
                    self.numTurns += 1
                    self.agent.playerTurns += 1

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            # Check if game Over
            if self.endCondition():
                return self.endGame(reward)
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, reward, done, truncated, {}

    def endGame(self, reward):
        while True:
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)

            if self.game.gameState.currState == "OVER":
                break
        
        reward = self.agent.victoryPoints

        return None, reward, True, False, {}
