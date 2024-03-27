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


class CatanBaseEnv(gym.Env):
    """
    Full Catan game with full action and state space (no player trades)
    """
    def __init__(self, customBoard=None, players=None, trading=False):
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

        self.trading = trading

        self.getActionMask = getActionMask
        self.getObservation = getObservationSimplified

        if players == None:
            self.players = [AgentRandom2("P0", 0, playerTrading=trading),
                            AgentRandom2("P1", 1, playerTrading=trading),
                            AgentRandom2("P2", 2, playerTrading=trading),
                            AgentRandom2("P3", 3, playerTrading=trading)]
        else:
            self.players = players

    def reset(self, seed=None):
        """
        Setup new game, cycle through actions till agents turn, return current state and set action mask
        """
        for player in self.players:
            player:Player = player.__init__(player.name, player.seatNumber, playerTrading=self.trading) 
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
    def __init__(self, customBoard=None, players=None, trading=False):
        super(CatanEnv, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 1
        self.loseRewardAmount = -1
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
            # Setup Rewards
        self.setupReward = True
        self.setupRewardMultiplier = 1
            # Speed Rewards
        self.winSpeedReward = False
        self.maxNumTurnsToWin = 100
            # Trading Rewards
        self.bankTradeReward = False
        self.bankTradeRewardMultiplier = 0.75
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = False
        self.denseRewardMultiplier = 1

        # Settings for Setup training
        self.observation_space = spaces.Box(low=lowerBoundsSimplified, high=upperBoundsSimplified, dtype=np.int64)
        self.getActionMask = getActionMask
        self.getObservation = getObservationSimplified

    
    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0
        self.setup1Production = listm([0, 0, 0, 0, 0, 0])
        self.setup1TradeRates = listm([4, 4, 4, 4, 4])
        return super(CatanEnv, self).reset()

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
            # For first placement, we want to get Ore+Wheat
            if actionObj.type == 'BuildSettlement' and prevState == 'START1A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                self.setup1Production = resourceProduction
                self.setup1TradeRates = self.agent.tradeRates

                # Get node value
                value = getNodeValue(self.game.gameState.boardNodes[actionObj.position], self.game.gameState)  
                reward += value
                
            # For second placement, we want Road - wood+brick
            elif actionObj.type == 'BuildSettlement' and prevState == 'START2A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                setup2Production = resourceProduction - self.setup1Production

                # Get node value
                value = getNodeValue(self.game.gameState.boardNodes[actionObj.position], self.game.gameState)  
                reward += value
                

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
        if wonGame:
            GAME_RESULTS.append(1)
            if self.winReward:
                reward += self.winRewardAmount
        else:
            GAME_RESULTS.append(0)
            if self.winReward:
                reward += self.loseRewardAmount

        return None, reward, True, False, {}
        

class SelfPlayUniform(CatanEnv):
    """
    When threshold reached updated all opponents to current model. 
    """
    def __init__(self):
        super(SelfPlayUniform, self).__init__()

        # Load starting opponent model
        self.opponentModel = MaskablePPO.load('DeepLearning/Models/BaselineSelfPlay/BaselineSelfPlay.zip')
    
    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_UNIFORM"] == "True":
            # Get name of model to update to
            modelName = os.environ["MODEL_NAME"]
            self.opponentModel.set_parameters(f"DeepLearning/Models/SelfPlayUniform/{modelName}")
            os.environ["UPDATE_MODELS_UNIFORM"] = "False"

        self.game = CreateGame([AgentRandom2("P0", 0),
                                AgentModel("P1", 1, self.opponentModel),
                                AgentModel("P2", 2, self.opponentModel),
                                AgentModel("P3", 3, self.opponentModel)])
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


class SelfPlayDistribution(CatanEnv):
    """
    When threshold reached update oppenents to [most recent, random pick, random pick, random pick]
    """
    def __init__(self):
        super(SelfPlayDistribution, self).__init__()

        # Load starting opponent model
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Models/BaselineSelfPlay/BaselineSelfPlay.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Models/BaselineSelfPlay/BaselineSelfPlay.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Models/BaselineSelfPlay/BaselineSelfPlay.zip')
    
    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_DIST"] == "True":
            modelName1 = os.environ["MODEL_1_NAME"]
            modelName2 = os.environ["MODEL_2_NAME"]
            modelName3 = os.environ["MODEL_3_NAME"]
            self.opponentModel1.set_parameters(f"DeepLearning/Models/SelfPlayDistribution/{modelName1}")
            self.opponentModel1.set_parameters(f"DeepLearning/Models/SelfPlayDistribution/{modelName2}")
            self.opponentModel1.set_parameters(f"DeepLearning/Models/SelfPlayDistribution/{modelName3}")
            os.environ["UPDATE_MODELS_DIST"] = "False"

        self.game = CreateGame([AgentRandom2("P0", 0),
                                AgentModel("P1", 1, self.opponentModel1),
                                AgentModel("P2", 2, self.opponentModel2),
                                AgentModel("P3", 3, self.opponentModel3)])
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


class CatanTradingEnv(CatanBaseEnv):

    """
    Full Catan game with full action and state space
    """
    def __init__(self, customBoard=None, players=None, trading=True):
        super(CatanTradingEnv, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 0
        self.loseRewardAmount = -40
        self.vpActionReward = True # Actions that directly give vp
        self.vpActionRewardMultiplier = 1
            # Setup Rewards
        self.setupReward = False
        self.setupRewardMultiplier = 1
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = False
        self.denseRewardMultiplier = 1
            # Trading Rewards (Accepting offer is treated same as bank trade)
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 0.1
        self.playerTradeReward = True
        self.playerTradeRewardMultiplier = 0.1

        self.getActionMask = getActionMaskTrading
        self.getObservation = getObservationTrading

        self.action_space = spaces.Discrete(566)
        self.observation_space = spaces.Box(low=lowerBoundsTrading, high=upperBoundsTrading, dtype=np.int64) 

        # Track number of trade offers per turn
        self.tradesThisTurn = 0

    
    def reset(self, seed=None):
        self.numTurns = 0
        self.turnsFirstSettlement = 0
        self.checkForTradeResult = False
        self.resourcesBeforeTradeOffer = None
        self.setup1Production = None
        self.setup1TradeRates = None
        return super(CatanTradingEnv, self).reset()

    def endCondition(self) -> bool:
        if self.numTurns >= 20 or self.game.gameState.currState =="OVER":
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


        if (self.bankTradeReward or self.playerTradeReward) and prevState[:5] != "START":
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
            self.agent.tradeCount = 0
        elif actionObj.type == "MakeTradeOffer":
            self.agent.tradeCount += 1

        if self.bankTradeReward:
            if actionObj.type == "BankTradeOffer" or actionObj.type == "AcceptTradeOffer":
                # print(f"Trade: {actionObj.getString()}\nResources: {self.agent.resources[:5]}\n\n")
                # DisplayImage(self.game.gameState)
                canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
                # Trades which allow us to build
                if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                    reward += 2 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == False and canBuildCityAfter == True:
                    reward += 2 * self.bankTradeRewardMultiplier
                #     # only in situations where we want to build road - haven't built 1st settlement or don't have longest road
                # if (canBuildRoadBefore == False and canBuildRoadAfter == True) and ((len(self.agent.settlements) <= 2 and not self.game.gameState.GetPossibleSettlements(self.agent)) or (len(self.agent.settlements) > 2 and self.game.gameState.longestRoadPlayer != self.agent.seatNumber)):
                #     reward += 1 * self.bankTradeRewardMultiplier
                # if (canBuyDevCardBefore == False and canBuyDevCardAfter == True) and ((len(self.agent.settlements) + len(self.agent.settlements)) > 2 and len(self.agent.cities) > 0):
                #     reward += 2 * self.bankTradeRewardMultiplier
                # Trades which get rid of resources for possible Builds
                if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                    reward -= 2 * self.bankTradeRewardMultiplier
                if canBuildCityBefore == True and canBuildCityAfter == False:
                    reward -= 2 * self.bankTradeRewardMultiplier
                # # Trade for no direct benefit
                # if canBuildSettlementAfter == False and canBuildRoadAfter == False and canBuildCityAfter == False and canBuyDevCardAfter == False:
                #     reward -= 2 * self.bankTradeRewardMultiplier

        if self.playerTradeReward:
            # If we just made an offer need to check next time around if offer was accepted
            if actionObj.type == "MakeTradeOffer":
                self.checkForTradeResult = True
                self.resourcesBeforeTradeOffer = self.agent.resources


        if self.vpActionReward:
            # if biggestArmyBefore == False and self.agent.biggestArmy == True:
            #     reward += 0.5 * self.vpActionRewardMultiplier
            # if biggestRoadBefore == False and self.agent.biggestRoad == True:  
            #     reward += 0.5 * self.vpActionRewardMultiplier
            # if self.agent.developmentCards[VICTORY_POINT_CARD_INDEX] - vpDevCardBefore == 1:
            #     reward += 0.25 * self.vpActionRewardMultiplier
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 1 * self.vpActionRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 1 * self.vpActionRewardMultiplier

        if self.denseRewards:
            if actionObj.type == 'BuyDevelopmentCard':
                # Shouldn't be buying dev Cards when we haven't built a settlement or if we haven't built a city
                if (len(self.agent.settlements) + len(self.agent.settlements)) <= 2 or len(self.agent.cities) == 0:
                    reward -= 0.5 * self.denseRewardMultiplier
                else:
                    reward += 3 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                possibleSettlementsAfter = self.game.gameState.GetPossibleSettlements(self.agent)
                # If theres was a spot to build first settlement and we build another road - punish
                if (len(self.agent.settlements) + len(self.agent.cities) <= 2) and possibleSettlementsBefore:
                    reward -= 0.5 * self.denseRewardMultiplier
                # If there was no spot to build 1st settlement then reward
                elif (len(self.agent.settlements) + len(self.agent.cities) <= 2) and len(possibleSettlementsBefore) == 0:
                    if possibleSettlementsAfter:
                        reward += 2 * self.denseRewardMultiplier
                    else:
                        reward += 1 * self.denseRewardMultiplier
                # If we've built a settlement and we don't have the longest road - reward
                if (len(self.agent.settlements) + len(self.agent.cities) >= 3) and self.game.gameState.longestRoadPlayer != self.agent.seatNumber:
                    reward += 1 * self.denseRewardMultiplier
                # If we do have longest road and theres possible settlements we shouldn't be building more roads
                if self.game.gameState.longestRoadPlayer == self.agent.seatNumber and possibleSettlementsAfter:
                    reward -= 0.5 * self.denseRewardMultiplier
            # Using dev card
            elif actionObj.type[:3] == 'Use':
                reward += 1

        if self.setupReward:
            if actionObj.type == 'BuildSettlement' and prevState == 'START1A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                self.setup1Production = resourceProduction
                self.setup1TradeRates = self.agent.tradeRates
                # Did we get a port
                if sum(self.setup1TradeRates) < 20:
                    for i in range(5):
                        # Matching 2:1 port with 3+ dot resource
                        if self.setup1TradeRates[i] == 2 and self.setup1Production[i] >= 3:
                            reward += 20 * self.setupRewardMultiplier
                        # 3:1 port with >= 6 dots
                        elif sum(self.setup1TradeRates) == 18 and sum(self.setup1Production) >= 6:
                            reward += 20 * self.setupRewardMultiplier
                # If we didn't get a port, need to get >= 10 dots
                else:
                    reward += sum(self.setup1Production) * self.setupRewardMultiplier
            elif actionObj.type == 'BuildSettlement' and prevState == 'START2A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                setup2Production = resourceProduction - self.setup1Production
                # Did we get a port
                if sum(self.agent.tradeRates) < sum(self.setup1TradeRates):
                    for i in range(5):
                        # Matching 2:1 port with 3+ dot resource
                        if self.agent.tradeRates[i] == 2 and setup2Production[i] >= 3:
                            reward += 20 * self.setupRewardMultiplier
                        # 3:1 port with >= 6 dots
                        elif  sum(self.setup1Production) >= 6:
                            reward += 20 * self.setupRewardMultiplier
                    else:
                        reward += sum(setup2Production) * self.setupRewardMultiplier


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
                    self.agent.tradeCount = 0

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            # Check if game Over
            if self.endCondition():
                return self.endGame(reward)
        
        # If we made an offer it will have either been accepted/rejected at this point
        if self.playerTradeReward:
            if self.checkForTradeResult:
                # If trade was accepted resources will have changed
                if self.resourcesBeforeTradeOffer != self.agent.resources:
                    canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                    canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                    canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                    canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
                    # Trades which allow us to build
                    if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                        reward += 2 * self.playerTradeRewardMultiplier
                    if canBuildCityBefore == False and canBuildCityAfter == True:
                        reward += 2 * self.playerTradeRewardMultiplier
                    # if (canBuildRoadBefore == False and canBuildRoadAfter == True) and ((len(self.agent.settlements) <= 2 and not self.game.gameState.GetPossibleSettlements(self.agent)) or (len(self.agent.settlements) > 2 and self.game.gameState.longestRoadPlayer != self.agent.seatNumber)):
                    #     reward += 1 * self.playerTradeRewardMultiplier
                    # if (canBuyDevCardBefore == False and canBuyDevCardAfter == True) and ((len(self.agent.settlements) + len(self.agent.settlements)) > 2 and len(self.agent.cities) > 0):
                    #     reward += 2 * self.playerTradeRewardMultiplier
                    # Trades which get rid of resources for possible Builds
                    if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                        reward -= 2 * self.playerTradeRewardMultiplier
                    if canBuildCityBefore == True and canBuildCityAfter == False:
                        reward -= 2 * self.playerTradeRewardMultiplier
                        # Punish more than bank trades since we could be helping other players
                    # if canBuildSettlementAfter == False and canBuildRoadAfter == False and canBuildCityAfter == False and canBuyDevCardAfter == False:
                    #     reward -= 2 * self.playerTradeRewardMultiplier
                self.checkForTradeResult = False

        
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
                reward += self.loseRewardAmount
        
        reward = self.agent.victoryPoints

        return None, reward, True, False, {}
    



class SelfPlayDistTradingEnv(CatanBaseEnv):

    """
    Full Catan game with full action and state space
    """
    def __init__(self, customBoard=None, players=None):
        super(SelfPlayDistTradingEnv, self).__init__(customBoard=customBoard, players=players, trading=True)

        # Reward settings
        self.winReward = True
        self.winRewardAmount = 100
        self.loseRewardAmount = -100
        self.vpActionReward = True # Actions that directly give vp
        self.vpActionRewardMultiplier = 5
            # Setup Rewards
        self.setupReward = False
        self.setupRewardMultiplier = 1
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = True
        self.denseRewardMultiplier = 0.5
            # Trading Rewards (Accepting offer is treated same as bank trade)
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 0.25
        self.playerTradeReward = True
        self.playerTradeRewardMultiplier = 0.25

        self.getActionMask = getActionMaskTrading
        self.getObservation = getObservationSimplified

        self.action_space = spaces.Discrete(566)

        # Track number of trade offers per turn
        self.tradesThisTurn = 0

        # Load starting opponent model
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Models/BaselineTradingModel.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Models/BaselineTradingModel.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Models/BaselineTradingModel.zip')

    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0
        self.checkForTradeResult = False
        self.resourcesBeforeTradeOffer = None
        self.setup1Production = None
        self.setup1TradeRates = None

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_DIST"] == "True":
            modelName1 = os.environ["MODEL_1_NAME"]
            modelName2 = os.environ["MODEL_2_NAME"]
            modelName3 = os.environ["MODEL_3_NAME"]
            self.opponentModel1.set_parameters(f"DeepLearning/Models/SelfPlayDist_Trading/{modelName1}")
            self.opponentModel1.set_parameters(f"DeepLearning/Models/SelfPlayDist_Trading/{modelName2}")
            self.opponentModel1.set_parameters(f"DeepLearning/Models/SelfPlayDist_Trading/{modelName3}")
            os.environ["UPDATE_MODELS_DIST"] = "False"

        self.game = CreateGame([AgentRandom2("P0", 0, playerTrading=True),
                                AgentModel("P1", 1, self.opponentModel1, playerTrading=True),
                                AgentModel("P2", 2, self.opponentModel2, playerTrading=True),
                                AgentModel("P3", 3, self.opponentModel3, playerTrading=True)])
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


        if (self.bankTradeReward or self.playerTradeReward) and prevState[:5] != "START":
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
            self.agent.tradeCount = 0
        elif actionObj.type == "MakeTradeOffer":
            self.agent.tradeCount += 1

        if self.bankTradeReward:
            if actionObj.type == "BankTradeOffer" or actionObj.type == "AcceptTradeOffer":
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
                    reward += 0.25 * self.bankTradeRewardMultiplier
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
                    reward -= 1 * self.bankTradeRewardMultiplier

        if self.playerTradeReward:
            # If we just made an offer need to check next time around if offer was accepted
            if actionObj.type == "MakeTradeOffer":
                self.checkForTradeResult = True
                self.resourcesBeforeTradeOffer = self.agent.resources


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
                reward += 3 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                # If haven't built first settlement yet and road increases number of possible settlements
                if (len(self.agent.settlements) + len(self.agent.cities) <= 2) and len(possibleSettlementsBefore) == 0 and len(self.game.gameState.GetPossibleSettlements(self.agent)) > 0:
                    reward += 1
            # Using dev card
            elif actionObj.type[:3] == 'Use':
                reward += 1

        if self.setupReward:
            if actionObj.type == 'BuildSettlement' and prevState == 'START1A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                self.setup1Production = resourceProduction
                self.setup1TradeRates = self.agent.tradeRates
                # Did we get a port
                if sum(self.setup1TradeRates) < 20:
                    for i in range(5):
                        # Matching 2:1 port with 3+ dot resource
                        if self.setup1TradeRates[i] == 2 and self.setup1Production[i] >= 3:
                            reward += 20 * self.setupRewardMultiplier
                        # 3:1 port with >= 6 dots
                        elif sum(self.setup1TradeRates) == 18 and sum(self.setup1Production) >= 6:
                            reward += 20 * self.setupRewardMultiplier
                # If we didn't get a port, need to get >= 10 dots
                else:
                    reward += sum(self.setup1Production) * self.setupRewardMultiplier
            elif actionObj.type == 'BuildSettlement' and prevState == 'START2A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                setup2Production = resourceProduction - self.setup1Production
                # Did we get a port
                if sum(self.agent.tradeRates) < sum(self.setup1TradeRates):
                    for i in range(5):
                        # Matching 2:1 port with 3+ dot resource
                        if self.agent.tradeRates[i] == 2 and setup2Production[i] >= 3:
                            reward += 20 * self.setupRewardMultiplier
                        # 3:1 port with >= 6 dots
                        elif  sum(self.setup1Production) >= 6:
                            reward += 20 * self.setupRewardMultiplier
                    else:
                        reward += sum(setup2Production) * self.setupRewardMultiplier


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
                    self.agent.tradeCount = 0

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

            # Check if game Over
            if self.endCondition():
                return self.endGame(reward)
        
        # If we made an offer it will have either been accepted/rejected at this point
        if self.playerTradeReward:
            if self.checkForTradeResult:
                # If trade was accepted resources will have changed
                if self.resourcesBeforeTradeOffer != self.agent.resources:
                    canBuildSettlementAfter = self.agent.CanAfford(BuildSettlementAction.cost)
                    canBuildRoadAfter = self.agent.CanAfford(BuildRoadAction.cost)
                    canBuildCityAfter = self.agent.CanAfford(BuildCityAction.cost)
                    canBuyDevCardAfter = self.agent.CanAfford(BuyDevelopmentCardAction.cost)
                    # Trades which allow us to build
                    if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                        reward += 4 * self.playerTradeRewardMultiplier
                    if canBuildCityBefore == False and canBuildCityAfter == True:
                        reward += 4 * self.playerTradeRewardMultiplier
                    if canBuildRoadBefore == False and canBuildRoadAfter == True :
                        reward += 0.5 * self.playerTradeRewardMultiplier
                    if canBuyDevCardBefore == False and canBuyDevCardAfter == True:
                        reward += 1 * self.playerTradeRewardMultiplier
                    # Trades which get rid of resources for possible Builds
                    if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                        reward -= 4 * self.playerTradeRewardMultiplier
                    if canBuildCityBefore == True and canBuildCityAfter == False:
                        reward -= 4 * self.playerTradeRewardMultiplier
                        # If we could buy dev card before and now can't get anything
                    if canBuyDevCardBefore == True and canBuyDevCardAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuildRoadAfter == False:
                        reward -= 2 * self.playerTradeRewardMultiplier
                        # If could build road before and now can't get anything
                    if canBuildRoadBefore == True and canBuildRoadAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuyDevCardAfter == False:
                        reward -= 1 * self.playerTradeRewardMultiplier
                        # Punish more than bank trades since we could be helping other players
                    if canBuildSettlementAfter == False and canBuildRoadAfter == False and canBuildCityAfter == False and canBuyDevCardAfter == False:
                        reward -= 2 * self.playerTradeRewardMultiplier
                self.checkForTradeResult = False

        
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
                reward += self.loseRewardAmount

        return None, reward, True, False, {}