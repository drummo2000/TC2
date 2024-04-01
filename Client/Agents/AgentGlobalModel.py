from Game.CatanGame import GameState
from Game.CatanPlayer import Player
from Game.CatanAction import *
from Agents.AgentRandom2 import AgentRandom2
import random
from DeepLearning.PPO import MaskablePPO
import logging
from global_model import global_models


class BaseAgentModel(AgentRandom2):
    """
    The Agents here are not used for training, they are either used as opponents for training
    or for testing pretrained models
    """
    def __init__(self, name, seatNumber, playerTrading: bool=False, recordStats=False, jsettlersGame=False, model_key=None):

        super(BaseAgentModel, self).__init__(name, seatNumber, playerTrading=playerTrading, recordStats=recordStats, jsettlersGame=jsettlersGame)
        self.jsettlersGame = jsettlersGame
        self.model_key = model_key

    def getModelAction(self, game, possibleActions):
        """
        Uses model and env to get action
        """
        action_masks, indexActionDict = global_models[self.model_key].getActionMask(possibleActions)
        state = global_models[self.model_key].getObservation(game.gameState, self.seatNumber)
        action, _states = global_models[self.model_key].predict(state, action_masks=action_masks)
        actionObj = indexActionDict[action.item()]
        return actionObj
    
    def getRandomAction(self, game, possibleActions):
        """
        Used when model hasn't been trained on parts of game
        """
        randIndex = random.randint(0, len(possibleActions)-1)
        chosenAction = possibleActions[randIndex]
        if chosenAction.type == "MakeTradeOffer":
            self.tradeCount += 1
        return chosenAction


class AgentGlobalModel(BaseAgentModel):
    """
    Agent which uses passed model for all moves
    """
    def DoMove(self, game):

        # For JSettlers
        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            #raise Exception("\n\nReturning None Action - INVESTIGATE\n\n")
            return None
        
        possibleActions = self.GetPossibleActions(game.gameState)

        if len(possibleActions) == 1:
            actionObj = possibleActions[0]
        else:
            actionObj = self.getModelAction(game, possibleActions)

            if self.playerTrading and actionObj.type == "MakeTradeOffer":
                self.tradeCount += 1
        
        if actionObj and actionObj.type == "EndTurn":
            self.playerTurns += 1

        return actionObj