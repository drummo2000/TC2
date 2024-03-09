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


class CatanEnv(gym.Env):
    """
    Full Catan game with full action and state space (no player trades)
    """
    def __init__(self, customBoard=None):
        super(CatanEnv, self).__init__()

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

    def reset(self, seed=None):
        """
        Setup new game, cycle through actions till agents turn, return current state and set action mask
        """
        inGame = CreateGame([   AgentRandom2("P0", 0),
                                AgentRandom2("P1", 1),
                                AgentRandom2("P2", 2),
                                AgentRandom2("P3", 3)], self.customBoard)
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



class ChangingRewardEnv(CatanEnv):
    """
    Full Catan game with full action and state space (no player trades)
    Change the rewards when certain reward reached
    """
    def __init__(self, customBoard=None):
        super(ChangingRewardEnv, self).__init__()

        # Reward settings
        self.winReward = False
        self.winRewardAmount = 50
        self.loseRewardAmount = 0
        self.vpActionReward = False # Actions that directly give vp
        self.vpActionRewardMultiplier = 4
            # Setup Rewards
        self.setupReward = True
        self.setupRewardMultiplier = 0
    
    def reset(self, seed=None):
        if os.environ.get("VP_REWARDS") == "True":
            self.vpActionReward = True
        if os.environ.get("WIN_REWARDS") == "True":
            self.winReward = True
        return super(ChangingRewardEnv, self).reset()


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

        # Add Rewards
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
        
        if self.setupReward:
            if actionObj.type == 'BuildSettlement' and prevState == 'START2A':
                resourceProduction = listm([0, 0, 0, 0, 0, 0])
                for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                    resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                reward += sum(resourceProduction) * self.setupRewardMultiplier
                # Diversity
                diversity = sum(x != 0 for x in resourceProduction)
                if diversity == 5:
                    reward += 20
                elif diversity == 4:
                    reward += 10
                elif diversity == 3:
                    reward += -10
                elif diversity == 2:
                    reward += -20

                # Check if it gets a single 2 port
                # correctPorts = sum(self.agent.tradeRates) == 18
                # if correctPorts:
                #     reward += 10
            

        # Check if game Over
        if self.game.gameState.currState == "PLAY":
            if self.winReward:
                wonGame = self.game.gameState.winner == 0
                if wonGame:
                    reward += self.winRewardAmount
                else:
                    reward += self.loseRewardAmount
            return None, reward, True, truncated, {}
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "PLAY":
                if self.winReward:
                    wonGame = self.game.gameState.winner == 0
                    if wonGame:
                        reward += self.winRewardAmount
                    else:
                        reward += self.loseRewardAmount
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
        