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
from DeepLearning.GetObservation import getObservation, getSetupObservation, getSetupRandomObservation, lowerBounds, upperBounds, lowerBoundsSimplified, upperBoundsSimplified, getObservationSimplified, getNodeValue, getObservationTrading, lowerBoundsTrading, upperBoundsTrading
from DeepLearning.GetActionMask import getActionMask, getActionMaskTrading
from DeepLearning.PPO import MaskablePPO
from CatanData.GameStateViewer import SaveGameStateImage, DisplayImage
import time
from collections import deque
from DeepLearning.globals import GAME_RESULTS
from DeepLearning.Environments.CatanEnv import CatanBaseEnv


class WinRewardEnv(CatanBaseEnv):
    """
    Gives rewards for number of vp at end of game
    """
    def __init__(self, customBoard=None, players=None, trading=False, lowerBounds=None, upperBounds=None, getObservationFunction=None, getActionMaskFunction=None):
        super(WinRewardEnv, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = True
        self.winRewardAmount = 10
        self.loseRewardAmount = -10
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMaskFunction
        self.getObservation = getObservationFunction
    
    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0
        return super(WinRewardEnv, self).reset()

    def endCondition(self) -> bool:
        if self.game.gameState.currState == "OVER":
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
            self.agent.tradeCount = 0
        elif actionObj.type == "MakeTradeOffer":
            self.agent.tradeCount += 1
                
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
                    self.agent.tradeCount = 0

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

        wonGame = self.game.gameState.winner == 0
        if wonGame:
            if self.winReward:
                reward += self.winRewardAmount
        else:
            if self.winReward:
                reward += self.loseRewardAmount
        
        # reward = self.agent.victoryPoints

        return None, reward, True, False, {}
    

class FinalvpRewardEnv(CatanBaseEnv):
    """
    Gives rewards for number of vp at end of game
    """
    def __init__(self, customBoard=None, players=None, trading=False, lowerBounds=None, upperBounds=None, getObservationFunction=None, getActionMaskFunction=None):
        super(FinalvpRewardEnv, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 1
        self.loseRewardAmount = -1
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMaskFunction
        self.getObservation = getObservationFunction
    
    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0
        return super(FinalvpRewardEnv, self).reset()

    def endCondition(self) -> bool:
        if self.game.gameState.currState == "OVER":
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
            self.agent.tradeCount = 0
        elif actionObj.type == "MakeTradeOffer":
            self.agent.tradeCount += 1
                
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
                    self.agent.tradeCount = 0

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

        # wonGame = self.game.gameState.winner == 0
        # if wonGame:
        #     if self.winReward:
        #         reward += self.winRewardAmount
        # else:
        #     if self.winReward:
        #         reward += self.loseRewardAmount
        
        reward = self.agent.victoryPoints

        return None, reward, True, False, {}