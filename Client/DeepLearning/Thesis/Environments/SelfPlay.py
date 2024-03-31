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
from DeepLearning.Thesis.Observations.get_observation import getObservation, lowerBound, upperBound

class SelfPlayBase(CatanBaseEnv):

    def __init__(self, customBoard=None, players=None, trading=False):
        super(SelfPlayBase, self).__init__(customBoard=customBoard, players=players, trading=trading)

        # Reward settings
        self.winReward = True
        self.winRewardAmount = 10
        self.loseRewardAmount = -10

        # Settings for Setup training
        self.observation_space = spaces.Box(low=lowerBound, high=upperBound, dtype=np.int64)
        self.action_space = spaces.Discrete(486)
        # self.action_space = spaces.Discrete(566)
        self.getActionMask = getActionMask
        self.getObservation = getObservation

    
    def reset(self, seed=None):
        self.numTurns = 0
        return super(SelfPlayBase, self).reset()

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

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if actionObj.type == "EndTurn":
            self.numTurns += 1
            self.agent.playerTurns += 1
                

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
                reward += self.loseRewardAmount

        return None, reward, True, False, {}
  

class SelfPlayUniform(SelfPlayBase):
    """
    When threshold reached updated all opponents to current model. 
    """
    def __init__(self):
        super(SelfPlayUniform, self).__init__()

        # Load starting opponent model
        self.opponentModel = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
    
    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_UNIFORM"] == "True":
            # Get name of model to update to
            modelName = os.environ["MODEL_NAME"]
            self.opponentModel.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Uniform/{modelName}")
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


class SelfPlayDistribution(SelfPlayBase):
    """
    When threshold reached update oppenents to [most recent, random pick, random pick, random pick]
    """
    def __init__(self):
        super(SelfPlayDistribution, self).__init__()

        # Load starting opponent model
        self.opponentModel1 = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
        self.opponentModel2 = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
        self.opponentModel3 = MaskablePPO.load('DeepLearning/Thesis/Opponents/Models/BaselineSelfPlay.zip')
    
    def reset(self, seed=None):

        self.numTurns = 0
        self.turnsFirstSettlement = 0

        # Update opponents models if needed
        if os.environ["UPDATE_MODELS_DIST"] == "True":
            modelName1 = os.environ["MODEL_1_NAME"]
            modelName2 = os.environ["MODEL_2_NAME"]
            modelName3 = os.environ["MODEL_3_NAME"]
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Distribution/{modelName1}")
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Distribution/{modelName2}")
            self.opponentModel1.set_parameters(f"DeepLearning/Thesis/Opponents/Models/Distribution/{modelName3}")
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

