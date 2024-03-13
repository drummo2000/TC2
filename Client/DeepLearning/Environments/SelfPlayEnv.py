import gymnasium as gym
import numpy as np
from gymnasium import spaces
import dill as pickle
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
from DeepLearning.Environments.CatanEnv import CatanBaseEnv
from Agents.AgentModel import AgentMultiModel, AgentModel


class SelfPlayEnv(CatanBaseEnv):
    """
    Self Play Environment which replaces opponents network every when threshold hit (timesteps/winrate/reward)
    """
    def __init__(self, customBoard=None):
        super(SelfPlayEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64) 

        self.getActionMask = getActionMask
        self.getObservation = getObservation

        self.opponentModel = MaskablePPO.load('DeepLearning/SelfPlayModels/UntrainedModel.zip')

    def reset(self, seed=None):
        """
        Check if opponenents models need to be updated and update if needed
        Setup new game, cycle through actions till agents turn, return current state and set action mask
        """
        if os.environ["UPDATE_MODELS"] == "True":
            modelName = os.environ["MODEL_NAME"]
            self.opponentModel.set_parameters(f"DeepLearning/SelfPlayModels/{modelName}.zip")
            os.environ["UPDATE_MODELS"] = "False"

        self.game = CreateGame([AgentRandom2("P0", 0),
                             AgentModel("P1", 1, self.opponentModel),
                             AgentModel("P2", 2, self.opponentModel),
                             AgentModel("P3", 3, self.opponentModel)], self.customBoard)
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
        if actionObj.type == 'BuildSettlement':
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





class SelfPlaySetupDotTotalEnv(CatanBaseEnv):
    """
    Self Play Environment which replaces opponents network every when threshold hit (timesteps/winrate/reward)
    All models use SetupOnlyDotTotal100k model for setupPhase
    Rewards: After an action gains a vp
    """
    def __init__(self, customBoard=None):
        super(SelfPlaySetupDotTotalEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64) 

        self.getActionMask = getActionMask
        self.getObservation = getObservation

        self.opponentModel = MaskablePPO.load('DeepLearning/Models/UntrainedModel.zip')
        self.setupModel = MaskablePPO.load('DeepLearning/Models/SetupOnly_DotTotal_100k.zip')

    def reset(self, seed=None):
        """
        Check if opponenents models need to be updated and update if needed
        Setup new game, cycle through actions till agents turn, return current state and set action mask
        """
        if os.environ["UPDATE_MODELS"] == "True":
            modelName = os.environ["MODEL_NAME"]
            self.opponentModel.set_parameters(f"DeepLearning/SelfPlayModels/{modelName}.zip")
            os.environ["UPDATE_MODELS"] = "False"

        self.game = CreateGame([AgentRandom2("P0", 0),
                             AgentMultiModel("P1", 1, self.opponentModel, self.setupModel, fullSetup=False),
                             AgentMultiModel("P2", 2, self.opponentModel, self.setupModel, fullSetup=False),
                             AgentMultiModel("P3", 3, self.opponentModel, self.setupModel, fullSetup=False)], self.customBoard)
        # self.game = pickle.loads(pickle.dumps(inGame, -1))
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
        if actionObj.type == 'BuildSettlement':
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