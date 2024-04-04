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
from DeepLearning.Thesis.get_observation import getObservation, lowerBound, upperBound


class WinRewardEnv(CatanBaseEnv):

    def __init__(self, customBoard=None, players=None, trading=False):
        super(WinRewardEnv, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = True
        self.winRewardAmount = 10
        self.loseRewardAmount = -10
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1

        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMask
        self.getObservation = getObservation
    
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




class VpActionRewardEnv(CatanBaseEnv):
    """
    Gives rewards for vp actions
    """
    def __init__(self, customBoard=None, players=None, trading=False, lowerBounds=None, upperBounds=None, getObservationFunction=None, getActionMaskFunction=None):
        super(VpActionRewardEnv, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 1
        self.loseRewardAmount = -1
        self.vpActionReward = True # Actions that directly give vp
        self.vpActionRewardMultiplier = 1

        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMaskFunction
        self.getObservation = getObservationFunction
    
    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0
        return super(VpActionRewardEnv, self).reset()

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
        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        vpDevCardBefore = self.agent.developmentCards[VICTORY_POINT_CARD_INDEX]
        prevState = self.game.gameState.currState

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1
            self.agent.tradeCount = 0
        elif actionObj.type == "MakeTradeOffer":
            self.agent.tradeCount += 1

        if self.vpActionReward:
            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                reward += 2 * self.vpActionRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:  
                reward += 2 * self.vpActionRewardMultiplier
            if self.agent.developmentCards[VICTORY_POINT_CARD_INDEX] - vpDevCardBefore == 1:
                reward += 1 * self.vpActionRewardMultiplier
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 1 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 1 * self.vpActionRewardMultiplier
                
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
        return None, reward, True, False, {}




class DenseRewardEnv(CatanBaseEnv):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """
    def __init__(self, customBoard=None, players=None, trading=False):
        super(DenseRewardEnv, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 1
        self.loseRewardAmount = -1
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
        return super(DenseRewardEnv, self).reset()

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
        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        vpDevCardBefore = self.agent.developmentCards[VICTORY_POINT_CARD_INDEX]
        prevState = self.game.gameState.currState

        if self.bankTradeReward and prevState[:5] != "START":
            possibleSettlementsBefore = self.game.gameState.GetPossibleSettlements(self.agent)
            canBuildSettlementBefore = possibleSettlementsBefore and self.agent.HavePiece(g_pieces.index('SETTLEMENTS')) and self.agent.CanAfford(BuildSettlementAction.cost)
            canBuildCityBefore = self.agent.settlements and self.agent.CanAfford(BuildCityAction.cost)
            canBuyDevCardBefore = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
            canBuildRoadBefore = self.game.gameState.GetPossibleRoads(self.agent) and self.agent.HavePiece(g_pieces.index('ROADS')) and self.agent.CanAfford(BuildRoadAction.cost)

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1

        if self.bankTradeReward:
            if actionObj.type == "BankTradeOffer":
                canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
                # Trades which allow us to build
                if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                    reward += 1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == False and canBuildCityAfter == True:
                    reward += 1 * self.bankTradeRewardMultiplier
                if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == True and canBuildCityAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier

        if self.vpActionReward:
            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                reward += 2 * self.vpActionRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:  
                reward += 2 * self.vpActionRewardMultiplier
            if self.agent.developmentCards[VICTORY_POINT_CARD_INDEX] - vpDevCardBefore == 1:
                reward += 1 * self.vpActionRewardMultiplier
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 1 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 1 * self.vpActionRewardMultiplier

        if self.denseRewards:
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuyDevelopmentCard':
                reward += 1 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                reward += 1 * self.denseRewardMultiplier
            # Using dev card
            # elif actionObj.type[:3] == 'Use':
            #     reward += 1
            # if biggestArmyBefore == False and self.agent.biggestArmy == True:
            #     reward += 5 * self.vpActionRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:  
                reward += 20 * self.vpActionRewardMultiplier

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
        wonGame = self.game.gameState.winner == 0
        if wonGame:
            if self.winReward:
                reward += self.winRewardAmount
        else:
            if self.winReward:
                reward += self.loseRewardAmount

        return None, reward, True, False, {}