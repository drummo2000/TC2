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
from DeepLearning.GetObservation import getInputState, getSetupInputState, getSetupRandomInputState, lowerBounds, upperBounds, setupRandomLowerBounds, setupRandomUpperBounds
from DeepLearning.GetActionMask import getActionMask, getSetupActionMask
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from DeepLearning.Environments.CatanEnv import CatanEnv
from Agents.AgentModel import AgentSelfPlayOpp


class SelfPlayEnv(CatanEnv):
    """
    Self Play Environment which replaces opponents network every when threshold hit (timesteps/winrate/reward)
    """
    def __init__(self, customBoard=None):
        super(SelfPlayEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64) 

        self.getActionMaskFunc = getActionMask
        self.getInputStateFunc = getInputState

        self.opponentModel = MaskablePPO.load('DeepLearning/Models/UntrainedModel.zip')

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
                             AgentSelfPlayOpp("P1", 1, self.opponentModel, getActionMask, getInputState),
                             AgentSelfPlayOpp("P2", 2, self.opponentModel, getActionMask, getInputState),
                             AgentSelfPlayOpp("P3", 3, self.opponentModel, getActionMask, getInputState)], self.customBoard)
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
        self.action_mask, self.indexActionDict = self.getActionMaskFunc(possibleActions)
        observation = self.getInputStateFunc(self.game.gameState)

        return observation, {}


    def step(self, action):
        """
        Accepts action index as argument, applies action, cycles through to players next turn, 
        gets observation and action mask for turn
        """
        truncated = False
        done = False

        prevVP = self.agent.victoryPoints

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        # get reward
        reward = self.agent.victoryPoints - prevVP

        if self.game.gameState.currState == "OVER":
            return None, reward, True, truncated, {"ActionMask": None}
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "OVER":
                return None, reward, True, truncated, {"ActionMask": None}
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMaskFunc(possibleActions)
        observation = self.getInputStateFunc(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, reward, done, truncated, {}
