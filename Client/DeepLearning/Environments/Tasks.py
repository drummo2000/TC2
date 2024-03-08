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
from DeepLearning.Environments.CatanEnv import CatanEnv


class FirstSettlementEnv(CatanEnv):
    """
    Game finishes when agent places first settlement (after setup), reward is least amount of moves to build it.
    """
    
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
        self.turns = 0
        self.numSettlementsBuilt = 0

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
        maximumTurns = 50

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)
        
        if actionObj.type == "EndTurn":
            self.turns += 1
        elif actionObj.type == "BuildSettlement":
            self.numSettlementsBuilt += 1

        # If 3rd settlement has been built finish game
        if self.numSettlementsBuilt > 2:
            reward += maximumTurns - self.turns
            return None, reward, True, truncated, {}
        
        if self.game.gameState.currState == "OVER":
            reward += maximumTurns - self.turns
            return None, reward, True, truncated, {}

        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "OVER":
                reward += maximumTurns - self.turns
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




class FirstCityEnv(CatanEnv):
    """
    Game finishes when agent places first city, reward is least amount of moves to build it.
    """
    
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
        self.turns = 0
        self.numCitiesBuilt = 0

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
        maximumTurns = 50

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)
        
        if actionObj.type == "EndTurn":
            self.turns += 1

        # If City built finish game
        if actionObj.type == "BuildCity":
            reward += maximumTurns - self.turns
            return None, reward, True, truncated, {}
        
        if self.game.gameState.currState == "OVER":
            reward += maximumTurns - self.turns
            return None, reward, True, truncated, {}

        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "OVER":
                reward += maximumTurns - self.turns
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
    



class SetupOnlyFirstCityEnv(CatanEnv):
    """
    Model Chooses placement actions and is trained on them, then pretrained model passed plays for rest of actions
    Reward: 50 - num turns to build first settlement
    Action Space: full
    State Space: full simplified
    """

    def __init__(self, trainedModel: MaskablePPO, customBoard=None):
        super(SetupOnlyFirstCityEnv, self).__init__()
        
        # self.agentModel = AgentModel("P0", 0, model=trainedModel)
        self.model = trainedModel
        self.maximumTurns = 50


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

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        if self.game.gameState.currState == "PLAY":
            return self.end_with_model()
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "PLAY":
                return self.end_with_model()
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, 0, done, truncated, {}
    
    # Simulate until agent has built City using passed pretrained model and return final reward
    def end_with_model(self):
        turns = 0
        while True:
            currPlayer = self.game.gameState.players[self.game.gameState.currPlayer]

            if currPlayer.seatNumber == 0:
                actionObj = self.modelDoMove(self.game)
                if actionObj.type == "EndTurn":
                    turns += 1
                elif actionObj.type == "BuildCity":
                    actionObj.ApplyAction(self.game.gameState)
                    break
            else:
                actionObj = currPlayer.DoMove(self.game)
            actionObj.ApplyAction(self.game.gameState)

            if self.game.gameState.currState == "OVER":
                break
        
        reward = self.maximumTurns - turns
        return None, reward, True, False, {}
    

    def modelDoMove(self, game):
        possibleActions = self.game.gameState.players[0].GetPossibleActions(game.gameState)
        if len(possibleActions) == 1:
            return possibleActions[0]

        action_masks, indexActionDict = self.model.getActionMask(possibleActions)
        state = self.model.getObservation(game.gameState)
        action, _states = self.model.predict(state, action_masks=action_masks)
        actionObj = indexActionDict[action.item()]
        return actionObj
