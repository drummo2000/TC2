import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Agents.AgentRandom2 import AgentRandom2
from CatanData.GameStateViewer import SaveGameStateImage
from DeepLearning.GetObservation import getSetupRandomObservation, getSetupRandomWithRoadsObservation, setupRandomLowerBounds, setupRandomUpperBounds, setupRandomWithRoadsLowerBounds, setupRandomWithRoadsUpperBounds
from DeepLearning.GetActionMask import getActionMask, getSetupActionMask, getSetupWithRoadsActionMask
from DeepLearning.Environments.CatanEnv import CatanEnv

# Takes in list of production for each dice number and returns weighted sum
def getProductionReward(productionDict: dict) -> int:
    reward = 0
    for diceNumber, list in productionDict.items():
        reward += numberDotsMapping[diceNumber] * sum(list)
    return reward



class SetupOnlyEnv(CatanEnv):
    """
    Game ends after setup phase
    Observation: all node info for each node (~1000?)
    Action space: 54 length for each node
    Reward: dot total from chosen node. 
    """
    phases = ['START1A', 'START2A']

    def __init__(self, lowerBounds, upperBounds, getActionMask, getObservation, customBoard=None):
        super(SetupOnlyEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64)

        # functions for getting actionMask/observation
        self.getActionMask = getActionMask
        self.getObservation = getObservation

        self.lastReward = 0

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

        self.lastReward = 0

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

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        # Get reward for action
        reward = getProductionReward(self.agent.diceProduction) - self.lastReward
        self.lastReward = reward
        adjReward = reward - 0

        if self.game.gameState.currState == "PLAY":
            return None, adjReward, True, truncated, {"ActionMask": None}
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0 or self.game.gameState.currState == "START1B" or self.game.gameState.currState == "START2B":
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "PLAY":
                return None, adjReward, True, truncated, {"ActionMask": None}
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, adjReward, done, truncated, {}



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################



class SetupRandomEnv(CatanEnv):
    """
    Model Chooses placement actions, then random actions for rest of the game
    Reward: +3 win, -1 loss
    Action Space: 54
    State Space: All node info
    """

    phases = ['START1A', 'START2A'] # which phases the model trained on this env should be used for

    def __init__(self, customBoard=None, opponents=None):
        super(SetupRandomEnv, self).__init__()

        self.action_space = spaces.Discrete(54)
        self.observation_space = spaces.Box(low=setupRandomLowerBounds, high=setupRandomUpperBounds, dtype=np.int64)

        # functions for getting actionMask/observation
        self.getActionMask = getSetupActionMask
        self.getObservation = getSetupRandomObservation

        self.opponents = None



    def reset(self, seed=None):
        """
        Setup new game, cycle through actions till agents turn, return current state and set action mask
        """
        if self.opponents:
            inGame = CreateGame(*self.opponents, self.customBoard)
        else:
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

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)


        if self.game.gameState.currState == "PLAY":
            return self.end_with_random()
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0 or self.game.gameState.currState == "START1B" or self.game.gameState.currState == "START2B":
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "PLAY":
                return self.end_with_random()
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, 0, done, truncated, {}
    
    # Simulate rest of the game using random actions and return reward
    def end_with_random(self):

        while True:
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]

            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)

            if self.game.gameState.currState == "OVER":
                break

        reward = 0
        if self.game.gameState.winner == 0:
            reward = 3
        else:
            reward = -1
        return None, reward, True, False, {}


######################################################################################################################################################
######################################################################################################################################################
    

class SetupRandomWithRoadsEnv(SetupRandomEnv):
    """
    Same as SetupRandom but including road placement
    Reward: See Parent
    Action Space: 54 + 72
    State Space: All node info + road info
    """

    phases = ['START1A', 'START1B', 'START2A', 'START2B']

    def __init__(self, customBoard=None, opponents=None):
        super(SetupRandomWithRoadsEnv, self).__init__(opponents=opponents)

        self.action_space = spaces.Discrete(54+72)
        self.observation_space = spaces.Box(low=setupRandomWithRoadsLowerBounds, high=setupRandomWithRoadsUpperBounds, dtype=np.int64)

        # tions for getting actionMask/observation
        self.getActionMask = getSetupWithRoadsActionMask
        self.getObservation = getSetupRandomWithRoadsObservation
    
    def step(self, action):
        """
        Accepts action index as argument, applies action, cycles through to players next turn, 
        gets observation and action mask for turn
        """
        truncated = False
        done = False

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)


        if self.game.gameState.currState == "PLAY":
            return self.end_with_random()
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "PLAY":
                return self.end_with_random()
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, 0, done, truncated, {}