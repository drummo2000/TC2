import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Game.CatanPlayer import Player
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentModel import AgentModel
from DeepLearning.GetActionMask import getActionMask, getActionMaskTrading
from DeepLearning.PPO import MaskablePPO
from DeepLearning.globals import GAME_RESULTS
from DeepLearning.Environments.CatanEnv import CatanBaseEnv
from DeepLearning.Thesis.Observations.get_observation_full import getObservationFull, lowerBound, upperBound


class SelfPlayTrading(CatanBaseEnv):

    """
    Full Catan game with full action and state space
    """
    def __init__(self, customBoard=None, players=None):
        super(SelfPlayTrading, self).__init__(customBoard=customBoard, players=players, trading=True)

        # Reward settings
        self.winReward = True
        self.winRewardAmount = 50
            # Dense Rewards - Building roads/Buying dev cards/steeling resource
        self.denseRewards = True
        self.denseRewardMultiplier = 1
            # Trading Rewards (Accepting offer is treated same as bank trade)
        self.bankTradeReward = True
        self.bankTradeRewardMultiplier = 1
        self.playerTradeReward = True
        self.playerTradeRewardMultiplier = 1

        self.getActionMask = getActionMaskTrading
        self.getObservation = getObservationFull

        self.action_space = spaces.Discrete(566)
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)

        # Load starting opponent model
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Thesis/7.Trading/Models/DummyModel.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Thesis/7.Trading/Models/DummyModel.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Thesis/7.Trading/Models/DummyModel.zip')

    def reset(self, seed=None):

        self.checkForTradeResult = False
        self.resourcesBeforeTradeOffer = None
        self.numTurns = 0

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_DIST"] == "True":
            modelName1 = os.environ["MODEL_1_NAME"]
            modelName2 = os.environ["MODEL_2_NAME"]
            modelName3 = os.environ["MODEL_3_NAME"]
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/7.Trading/Models/SelfPlayTrading/{modelName1}")
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/7.Trading/Models/SelfPlayTrading/{modelName2}")
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/7.Trading/Models/SelfPlayTrading/{modelName3}")
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


        if prevState[:5] != "START":
            possibleSettlementsBefore = self.game.gameState.GetPossibleSettlements(self.agent)
            canBuildSettlementBefore = possibleSettlementsBefore and self.agent.HavePiece(g_pieces.index('SETTLEMENTS')) and self.agent.CanAfford(BuildSettlementAction.cost)
            canBuildCityBefore = self.agent.settlements and self.agent.CanAfford(BuildCityAction.cost)

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)


        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.tradeCount = 0
            self.agent.playerTurns += 1
        elif actionObj.type == "MakeTradeOffer":
            self.agent.tradeCount += 1

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

        if self.playerTradeReward:
            # If we just made an offer need to check next time around if offer was accepted
            if actionObj.type == "MakeTradeOffer":
                self.checkForTradeResult = True
                self.resourcesBeforeTradeOffer = self.agent.resources


        if self.denseRewards:
            if actionObj.type == 'BuildSettlement' and prevState[:5] != "START":
                reward += 10 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildCity':
                reward += 10 * self.denseRewardMultiplier
            elif actionObj.type == 'BuyDevelopmentCard':
                reward += 2 * self.denseRewardMultiplier
            elif actionObj.type == 'BuildRoad' and prevState[:5] != "START":
                reward += 1 * self.denseRewardMultiplier
            if biggestArmyBefore == False and self.agent.biggestArmy == True:
                reward += 10 * self.denseRewardMultiplier
            if biggestRoadBefore == False and self.agent.biggestRoad == True:  
                reward += 10 * self.denseRewardMultiplier


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
                    self.agent.playerTurns += 1

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
                    # Trades which get rid of resources for possible Builds
                    if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                        reward -= -2 * self.playerTradeRewardMultiplier
                    if canBuildCityBefore == True and canBuildCityAfter == False:
                        reward -= -2 * self.playerTradeRewardMultiplier
                    # Punish more than bank trades since we could be helping other players
                    if canBuildSettlementAfter == False and canBuildRoadAfter == False and canBuildCityAfter == False and canBuyDevCardAfter == False:
                        reward -= 1 * self.playerTradeRewardMultiplier
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
                reward += -5 * (10-self.agent.victoryPoints)

        return None, reward, True, False, {}