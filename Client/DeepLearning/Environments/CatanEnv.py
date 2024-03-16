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
from DeepLearning.GetObservation import getObservation, getSetupObservation, getSetupRandomObservation, lowerBounds, upperBounds, lowerBoundsSimplified, upperBoundsSimplified, getObservationSimplified
from DeepLearning.GetActionMask import getActionMask, getSetupActionMask
from DeepLearning.PPO import MaskablePPO
from CatanData.GameStateViewer import SaveGameStateImage, DisplayImage
import time


class CatanBaseEnv(gym.Env):
    """
    Full Catan game with full action and state space (no player trades)
    """
    def __init__(self, customBoard=None, players=None):
        super(CatanBaseEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBoundsSimplified, high=upperBoundsSimplified, dtype=np.int64) 

        self.game: Game = None
        # Used to fetch Action object from action index chosen by models
        self.indexActionDict = None
        self.players = []
        self.agent = None
        self.customBoard = customBoard
        self.num_envs = 1
        self.action_mask = None
        self.seed = 1

        self.getActionMask = getActionMask
        self.getObservation = getObservationSimplified

        if players == None:
            self.players = [AgentRandom2("P0", 0),
                            AgentRandom2("P1", 1),
                            AgentRandom2("P2", 2),
                            AgentRandom2("P3", 3)]
        else:
            self.players = players

    def reset(self, seed=None):
        """
        Setup new game, cycle through actions till agents turn, return current state and set action mask
        """
        for player in self.players:
            player:Player = player.__init__(player.name, player.seatNumber) 
        inGame = CreateGame(self.players, self.customBoard)
        self.game = pickle.loads(pickle.dumps(inGame, -1))
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

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        # get reward
        if biggestArmyBefore == False and self.agent.biggestArmy == True:
            reward += 2
        if biggestRoadBefore == False and self.agent.biggestRoad == True:  
            reward += 2
        if self.agent.developmentCards[VICTORY_POINT_CARD_INDEX] - vpDevCardBefore == 1:
            reward += 1
        if actionObj.type == 'BuildSettlement' and self.game.gameState.currState[:5] != 'START':
            reward += 1
        elif actionObj.type == 'BuildCity':
            reward += 1

        if self.game.gameState.currState == "OVER":
            return None, reward, True, truncated, {}
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "OVER":
                return None, reward, True, truncated, {}
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, reward, done, truncated, {}
    
    # Used by PPO algorithm
    def action_masks(self):
        return self.action_mask



class CatanEnv(CatanBaseEnv):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """
    def __init__(self, customBoard=None, players=None):
        super(CatanEnv, self).__init__(customBoard=customBoard, players=players)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 0
        self.loseRewardAmount = -50
        self.vpActionReward = True # Actions that directly give vp
        self.vpActionRewardMultiplier = 10
            # Setup Rewards
        self.setupReward = True
        self.setupRewardMultiplier = 0.75
            # Speed Rewards
        self.winSpeedReward = False
        self.maxNumTurnsToWin = 100
            # Trading Rewards
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 0.75
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = True
        self.denseRewardMultiplier = 1

    
    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0
        return super(CatanEnv, self).reset()

    def endCondition(self) -> bool:
        # Default
        # if self.game.gameState.currState == "OVER":
        #     return True
        # else:
        #     return False
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

        # print(actionObj.type)

        if actionObj.type == "EndTurn":
            self.numTurns += 1

        if self.bankTradeReward:
            if actionObj.type == "BankTradeOffer":
                # print(f"Trade: {actionObj.getString()}\nResources: {self.agent.resources[:5]}\n\n")
                # DisplayImage(self.game.gameState)
                canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
                # Trades which allow us to build
                if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                    # print("SettlementTRADE")
                    reward += 4 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == False and canBuildCityAfter == True:
                    # print("CityTRADE")
                    reward += 4 * self.bankTradeRewardMultiplier
                if canBuildRoadBefore == False and canBuildRoadAfter == True :
                    # print("RoadTRADE")
                    reward += 1 * self.bankTradeRewardMultiplier
                if canBuyDevCardBefore == False and canBuyDevCardAfter == True:
                    # print("DevCardTRADE")
                    reward += 2 * self.bankTradeRewardMultiplier
                # Trades which get rid of resources for possible Builds
                if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                    reward -= 4 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == True and canBuildCityAfter == False:
                    reward -= 4 * self.bankTradeRewardMultiplier
                    # If we could buy dev card before and now can't get anything
                if canBuyDevCardBefore == True and canBuyDevCardAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuildRoadAfter == False:
                    reward -= 2 * self.bankTradeRewardMultiplier
                    # If could build road before and now can't get anything
                if canBuildRoadBefore == True and canBuildRoadAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuyDevCardAfter == False:
                    reward -= 1 * self.bankTradeRewardMultiplier
                    # Small negative if we trade for no reason (Risk here that we stop long term planning of builds)
                if canBuildSettlementAfter == False and canBuildRoadAfter == False and canBuildCityAfter == False and canBuyDevCardAfter == False:
                    reward -= 0.25 * self.bankTradeRewardMultiplier


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
            if actionObj.type == 'BuyDevelopmentCard':
                reward += 4 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                # reward += 0.5 * self.denseRewardMultiplier
                # If before we have built 1st settlement we build another Road - punish
                if (len(self.agent.settlements) + len(self.agent.cities) <= 2) and possibleSettlementsBefore:
                    reward -= 2 * self.denseRewardMultiplier
                # If we increase number of possible settlements - reward
                elif (len(self.agent.settlements) + len(self.agent.cities) <= 2) and len(possibleSettlementsBefore) == 0:
                    reward += 1
            elif actionObj.type == 'PlaceRobber' and self.game.gameState.currState == "WAITING_FOR_CHOICE":
                reward += 0.5 * self.denseRewardMultiplier
            # Using dev card
            elif actionObj.type[:3] == 'Use':
                reward += 1


        
        if self.setupReward:
            if actionObj.type == 'BuildSettlement' and prevState == 'START2A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                reward += sum(resourceProduction) * self.setupRewardMultiplier
                # Diversity
                diversity = sum(x != 0 for x in resourceProduction)
                if diversity == 5:
                    reward += 7 * self.setupRewardMultiplier
                elif diversity == 4:
                    reward += 3 * self.setupRewardMultiplier


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
        if self.winReward:
            if wonGame:
                reward += self.winRewardAmount
            else:
                reward += self.loseRewardAmount

        return None, reward, True, False, {}
        