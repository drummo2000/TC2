import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Game.CatanPlayer import Player
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentModel import AgentModel
from Agents.AgentGlobalModel import AgentGlobalModel
import time
from collections import deque
from DeepLearning.globals import GAME_RESULTS
from DeepLearning.Environments.CatanEnv import CatanBaseEnv
from DeepLearning.Thesis.Setup.getActionMaskSetup import getSetupActionMask
from DeepLearning.Thesis.Setup.getObservationSetup import getObservationSetup, lowerBound, upperBound 
from global_model import global_models


class SetupRandom(CatanBaseEnv):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """
    def __init__(self, customBoard=None, players=None, trading=False):
        super(SetupRandom, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = True
        self.winRewardAmount = +10
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
        self.action_space = spaces.Discrete(126)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getSetupActionMask
        self.getObservation = getObservationSetup

    
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
        
        wonGame = self.game.gameState.winner == 0
        if wonGame:
            if self.winReward:
                reward += self.winRewardAmount
        else:
            if self.winReward:
                reward += self.loseRewardAmount

        return None, reward, True, False, {}


class SetupRandomSettlement(SetupRandom):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """

    def endGame(self, reward):
        while True:
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)

            if currPlayer.seatNumber == 0 and agentAction.type == "EndTurn":
                self.numTurns += 1

            if self.game.gameState.currState == "OVER" or (len(self.agent.settlements)+len(self.agent.cities)>=3):
                break
        
        if self.numTurns != 0:
            reward = 10 / self.numTurns
        else:
            reward = 10

        return None, reward, True, False, {}
    

class SetupAgentSettlement(SetupRandomSettlement):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """

    def __init__(self, customBoard=None, players=None, trading=False):
        super(SetupAgentSettlement, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = +10
        self.loseRewardAmount = -10
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
            # Trading Rewards
        self.bankTradeReward = False
        self.bankTradeRewardMultiplier = 1
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = False
        self.denseRewardMultiplier = 1

        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(126)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getSetupActionMask
        self.getObservation = getObservationSetup

    def reset(self, seed=None):
        self.numTurns = 0

        self.game = CreateGame([AgentGlobalModel("P0", 0, model_key="SelfPlayDense"),
                                AgentRandom2("P1", 1),
                                AgentRandom2("P2", 2),
                                AgentRandom2("P3", 3)])
        # self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through until agents turn
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, {}
    


class SetupAgentCities(SetupAgentSettlement):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """
    
    def endGame(self, reward):
        while True:
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)

            if self.game.gameState.currState == "OVER":
                break
        
        reward = len(self.agent.cities)

        return None, reward, True, False, {}


class SetupDiversity(CatanBaseEnv):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """
    def __init__(self, customBoard=None, players=None, trading=False):
        super(SetupDiversity, self).__init__(customBoard=customBoard, players=players, trading=trading)

        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(126)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getSetupActionMask
        self.getObservation = getObservationSetup

    
    def reset(self, seed=None):
        self.numTurns = 0
        return super(SetupDiversity, self).reset()

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
        resourceProduction = listm([0, 0, 0, 0, 0, 0])
        for diceNumber, resourceList in self.agent.diceProduction.items():
            resourceProduction += resourceList
        diversity = sum(x != 0 for x in resourceProduction[:-1])

        reward = diversity

        return None, reward, True, False, {}