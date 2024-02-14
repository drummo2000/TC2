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
from .GetObservation import getInputState, getSetupInputState, getSetupRandomInputState, lowerBounds, upperBounds, setupRandomLowerBounds, setupRandomUpperBounds
from .GetActionMask import getActionMask, getSetupActionMask
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


# Takes in list of production for each dice number and returns weighted sum
def getProductionReward(productionDict: dict) -> int:
    reward = 0
    for diceNumber, list in productionDict.items():
        reward += numberDotsMapping[diceNumber] * sum(list)
    return reward

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################



class CatanEnv(gym.Env):
    def __init__(self, customBoard=None):
        super(CatanEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64) 

        self.game: Game = None
        # Used to fetch Action object from action index chosen by models
        self.indexActionDict = None
        self.players = []
        self.agent = None
        self.customBoard = customBoard
        self.num_envs = 1
        self.action_mask = None
        self.seed = 1

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
        self.action_mask, self.indexActionDict = getActionMask(possibleActions)
        observation = getInputState(self.game.gameState)

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

        # get reward
        # winner = self.game.gameState.winner
        reward = self.agent.victoryPoints - 7
        # if winner == 0:
        #     reward = 3
        # elif winner == 1 or winner == 2 or winner == 3:
        #     reward = -1

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
        self.action_mask, self.indexActionDict = getActionMask(possibleActions)
        observation = getInputState(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, 0, done, truncated, {}
    
    # Used by PPO algorithm
    def action_masks(self):
        return self.action_mask



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################


    


class CatanSetupEnv(gym.Env):
    def __init__(self, customBoard=None):
        super(CatanSetupEnv, self).__init__()

        self.action_space = spaces.Discrete(54)
        self.observation_space = spaces.Box(shape=(54,), low=0, high=13, dtype=np.int64) 

        self.game: Game = None
        # Used to fetch Action object from action index chosen by models
        self.indexActionDict = None
        self.players = []
        self.agent = None
        self.lastReward = 0
        self.customBoard = customBoard
        self.num_envs = 1
        self.action_mask = None
        self.seed = 1

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
        self.action_mask, self.indexActionDict = getSetupActionMask(possibleActions)
        observation = getSetupInputState(self.game.gameState)

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
        self.action_mask, self.indexActionDict = getSetupActionMask(possibleActions)
        observation = getSetupInputState(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, adjReward, done, truncated, {}
    
    # Used by PPO algorithm
    def action_masks(self):
        return self.action_mask



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################



class CatanSetupRandomEnv(gym.Env):
    """
    Model Chooses placement actions, then random actions for rest of the game
    Reward: +3 win, -1 loss
    Action Space: 54
    State Space
    """

    def __init__(self, customBoard=None):
        super(CatanSetupRandomEnv, self).__init__()

        self.action_space = spaces.Discrete(54)
        self.observation_space = spaces.Box(low=setupRandomLowerBounds, high=setupRandomUpperBounds, dtype=np.int64) 

        self.game: Game = None
        # Used to fetch Action object from action index chosen by models
        self.indexActionDict = None
        self.players = []
        self.agent = None
        self.customBoard = customBoard
        self.num_envs = 1
        self.action_mask = None
        self.seed = 1

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
        self.action_mask, self.indexActionDict = getSetupActionMask(possibleActions)
        observation = getSetupRandomInputState(self.game.gameState)

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
        self.action_mask, self.indexActionDict = getSetupActionMask(possibleActions)
        observation = getSetupRandomInputState(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, 0, done, truncated, {}
    
    # Used by PPO algorithm
    def action_masks(self):
        return self.action_mask
    
    # Simulate rest of the game using random actions and return reward
    def end_with_random(self):
        if os.environ['SAVE_IMAGE'] == 'True':
            SaveGameStateImage(self.game.gameState, "setup_random.png")

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
######################################################################################################################################################
######################################################################################################################################################



class NoSetupEnv(gym.Env):
    """
    Separate trained model chooses setup actions, training on rest of game
    Action: full 486 (no player trades)
    State: full 2350
    Reward: +1 for win, -4 for loss
    """
    def __init__(self, setupModel: MaskablePPO, customBoard=None):
        super(NoSetupEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBounds, high=upperBounds, dtype=np.int64) 

        self.game: Game = None
        # Used to fetch Action object from action index chosen by models
        self.indexActionDict = None
        self.players = []
        self.agent = None
        self.customBoard = customBoard
        self.action_mask = None
        # Needed for MaskablePPO class
        self.num_envs = 1
        self.seed = 1

        self.setupModel = setupModel

    def reset(self, seed=None):
        """
        Setup new game, use SetupModel for setup moves, return state and action mask on agents first turn.
        """
        inGame = CreateGame([   AgentRandom2("P0", 0),
                                AgentRandom2("P1", 1),
                                AgentRandom2("P2", 2),
                                AgentRandom2("P3", 3)], self.customBoard)
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
                observation = getSetupRandomInputState(self.game.gameState)
                tempActionMask, tempIndexActionDict = getSetupActionMask(self.agent.GetPossibleActions(self.game.gameState))
                action, _states = self.setupModel.predict(observation, action_masks=tempActionMask)
                actionObj = tempIndexActionDict[action.item()]
                actionObj.ApplyAction(self.game.gameState)

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = getActionMask(possibleActions)
        observation = getInputState(self.game.gameState)

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
        self.action_mask, self.indexActionDict = getActionMask(possibleActions)
        observation = getInputState(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, 0, done, truncated, {}
    
    # Used by PPO algorithm
    def action_masks(self):
        return self.action_mask
    
    def getWinningReward(self):
        winner = self.game.gameState.winner
        if winner == 0:
            return 1
        else:
            return -4