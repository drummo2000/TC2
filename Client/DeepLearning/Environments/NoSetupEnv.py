import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
import os.path
import matplotlib.pyplot as plt
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Game.CatanPlayer import Player
from Agents.AgentRandom2 import AgentRandom2
from CatanData.GameStateViewer import SaveGameStateImage
from DeepLearning.GetObservation import getObservation, getSetupObservation, getSetupRandomObservation, lowerBounds, upperBounds, setupRandomLowerBounds, setupRandomUpperBounds
from DeepLearning.GetActionMask import getActionMask, getSetupActionMask
from DeepLearning.PPO import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from DeepLearning.Environments.CatanEnv import CatanEnv

class NoSetupEnv(CatanEnv):
    """
    Separate trained model chooses setup actions, training on rest of game
    Action: full 486 (no player trades)
    State: full 2350
    Reward: +1 for win, -3 for loss
    Opponents: Random Agents
    """
    
    def __init__(self, setupModel: MaskablePPO=None, customBoard=None):
        super(NoSetupEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64) 

        # functions for getting actionMask/observation
        self.getActionMask = getActionMask
        self.getObservation = getObservation

        self.setupModel = setupModel

    def reset(self, seed=None, players=None):
        """
        Setup new game, use SetupModel for setup moves, return state and action mask on agents first turn.
        """
        if players == None:
            players = [ AgentRandom2("P0", 0),
                        AgentRandom2("P1", 1),
                        AgentRandom2("P2", 2),
                        AgentRandom2("P3", 3)]
        inGame = CreateGame(players, self.customBoard)
        self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through setup phase and until its agents first regular turn (need to pass action/observable space used in setupModel training)
        tempActionMask = None
        tempIndexActionDict = None
        while True:
            currPlayer = self.players[self.game.gameState.currPlayer]
            if currPlayer.seatNumber != 0 or self.game.gameState.currState == 'START1B' or self.game.gameState.currState == 'START2B':
                agentAction = currPlayer.DoMove(self.game)
                agentAction.ApplyAction(self.game.gameState)
            else:
                if self.game.gameState.currState == 'PLAY':
                    break
                observation = getSetupRandomObservation(self.game.gameState)
                tempActionMask, tempIndexActionDict = getSetupActionMask(self.agent.GetPossibleActions(self.game.gameState))
                action, _states = self.setupModel.predict(observation, action_masks=tempActionMask)
                actionObj = tempIndexActionDict[action.item()]
                actionObj.ApplyAction(self.game.gameState)

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

        vpBefore = self.agent.victoryPoints

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        vpAfter = self.agent.victoryPoints

        #vpReward = vpAfter - vpBefore

        if self.game.gameState.currState == "OVER":
            return None, self.getWinningReward(), True, truncated, {"ActionMask": None}
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "OVER":
                return None, self.getWinningReward(), True, truncated, {"ActionMask": None}
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, 0, done, truncated, {}
    
    def getWinningReward(self):
        winner = self.game.gameState.winner
        if winner == 0:
            return 1
        else:
            return -3



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################



class NoSetupDenseRewardEnv(NoSetupEnv):
    """
    v1:
    Separate trained model chooses setup actions, training on rest of game
    Action: full 486 (no player trades)
    State: full 2350
    Reward: +100 for win, -50 for loss, reward for - BuildSettlement 10, BuildCity 10, BuyDevelopmentCard 5, BuildRoad 3, largest Army - 15, longest road 15
    """

    def step(self, action):
        """
        Accepts action index as argument, applies action, cycles through to players next turn, 
        gets observation and action mask for turn
        """
        truncated = False
        done = False
        denseReward = 0

        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if biggestArmyBefore == False and self.agent.biggestArmy == True:
            denseReward += 15
        if biggestRoadBefore == False and self.agent.biggestRoad == True:  
            denseReward += 15

        # Get reward
        if actionObj.type == 'BuildSettlement':
            denseReward += 10
        elif actionObj.type == 'BuildCity':
            denseReward += 10
        elif actionObj.type == 'BuyDevelopmentCard':
            denseReward += 5
        elif actionObj.type == 'BuildRoad':
            denseReward += 3


        if self.game.gameState.currState == "OVER":
            return None, self.getWinningReward() + denseReward, True, truncated, {"ActionMask": None}
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "OVER":
                return None, self.getWinningReward() + denseReward, True, truncated, {"ActionMask": None}
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, denseReward, done, truncated, {}
    
    def getWinningReward(self):
        winner = self.game.gameState.winner
        if winner == 0:
            return 100
        else:
            return -50