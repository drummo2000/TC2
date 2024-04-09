import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Game.CatanPlayer import Player
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentModel import AgentMultiModel
from Agents.AgentModel import AgentModel
from Agents.AgentGlobalModel import AgentGlobalModel
from DeepLearning.GetActionMask import getActionMask
from DeepLearning.PPO import MaskablePPO
from DeepLearning.globals import GAME_RESULTS
from DeepLearning.Environments.CatanEnv import CatanBaseEnv
from DeepLearning.Thesis.Observations.get_observation import getObservation, lowerBound, upperBound

class TurnLimitMultiModel(CatanBaseEnv):
    
    def __init__(self, setup_model_key, customBoard=None, players=None, trading=False):
        super(TurnLimitMultiModel, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = True
        self.winRewardAmount = 50
        self.loseRewardAmount = -100
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
            # Trading Rewards
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = True
        self.denseRewardMultiplier = 1

        # Settings for Setup training
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMask
        self.getObservation = getObservation

        self.model_key = setup_model_key

    def reset(self, seed=None, players=None):
        """
        Setup new game, use SetupModel for setup moves, return state and action mask on agents first turn.
        """
        if players == None:
            players = [ AgentGlobalModel("P0", 0, model_key=self.model_key),
                        AgentRandom2("P1", 1),
                        AgentRandom2("P2", 2),
                        AgentRandom2("P3", 3)]
        inGame = CreateGame(players, self.customBoard)
        self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through setup phase and until its agents first regular turn (need to pass action/observable space used in setupModel training)
        while True:
            currPlayer = self.players[self.game.gameState.currPlayer]
            if currPlayer.seatNumber != 0:
                agentAction = currPlayer.DoMove(self.game)
                agentAction.ApplyAction(self.game.gameState)
            else:
                if self.game.gameState.currState == 'PLAY':
                    break
                agentAction = currPlayer.DoMove(self.game)
                agentAction.ApplyAction(self.game.gameState)

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        self.numTurns = 0
        self.maxTurns = int(os.environ.get("TURN_LIMIT"))

        return observation, {}
    
    def endCondition(self) -> bool:
        if self.game.gameState.currState == "OVER" or self.numTurns >= self.maxTurns:
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
                if canBuildSettlementAfter == False and canBuildCityAfter == False and canBuildRoadAfter == False and canBuyDevCardAfter == False:
                    reward += -0.25 * self.bankTradeRewardMultiplier

        if self.denseRewards:
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuyDevelopmentCard':
                reward += 2 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                reward += 1 * self.denseRewardMultiplier
            # Using dev card
            # elif actionObj.type[:3] == 'Use':
            #     reward += 1
            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                reward += 10 * self.vpActionRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:  
                reward += 10 * self.vpActionRewardMultiplier

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
            GAME_RESULTS.append(1)
            if self.winReward:
                reward += self.winRewardAmount
        else:
            GAME_RESULTS.append(0)
            if self.winReward:
                reward += -5 * (10-self.agent.victoryPoints)

        return None, reward, True, False, {}
    

class TurnLimitMultiModelSettl(TurnLimitMultiModel):

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
                    reward += 2 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == False and canBuildCityAfter == True:
                    reward += 1 * self.bankTradeRewardMultiplier
                if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == True and canBuildCityAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier
                if canBuildSettlementAfter == False and canBuildCityAfter == False and canBuildRoadAfter == False and canBuyDevCardAfter == False:
                    reward += -0.25 * self.bankTradeRewardMultiplier

        if self.denseRewards:
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 15 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuyDevelopmentCard':
                reward += 2 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                reward += 1 * self.denseRewardMultiplier
            # Using dev card
            # elif actionObj.type[:3] == 'Use':
            #     reward += 1
            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                reward += 10 * self.vpActionRewardMultiplier
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
    

class TurnLimitMultiModelCity(TurnLimitMultiModel):

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
                    reward += 2 * self.bankTradeRewardMultiplier
                if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == True and canBuildCityAfter == False:
                    reward += -1 * self.bankTradeRewardMultiplier
                if canBuildSettlementAfter == False and canBuildCityAfter == False and canBuildRoadAfter == False and canBuyDevCardAfter == False:
                    reward += -0.25 * self.bankTradeRewardMultiplier

        if self.denseRewards:
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 10 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 15 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuyDevelopmentCard':
                reward += 4 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                reward += 1 * self.denseRewardMultiplier
            # Using dev card
            # elif actionObj.type[:3] == 'Use':
            #     reward += 1
            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                reward += 20 * self.vpActionRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:  
                reward += 10 * self.vpActionRewardMultiplier

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