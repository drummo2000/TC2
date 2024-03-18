from Game.CatanGame import GameState
from Game.CatanPlayer import Player
from Game.CatanAction import *
from Agents.AgentRandom2 import AgentRandom2
import random
from DeepLearning.PPO import MaskablePPO
import logging


class BaseAgentModel(AgentRandom2):
    """
    The Agents here are not used for training, they are either used as opponents for training
    or for testing pretrained models
    """
    def __init__(self, name, seatNumber, model: MaskablePPO, playerTrading: bool=False, recordStats=False):

        super(BaseAgentModel, self).__init__(name, seatNumber, playerTrading=playerTrading, recordStats=recordStats)
        self.model                  = model

    def getModelAction(self, game, possibleActions):
        """
        Uses model and env to get action
        """
        action_masks, indexActionDict = self.model.getActionMask(possibleActions)
        state = self.model.getObservation(game.gameState, self.seatNumber)
        action, _states = self.model.predict(state, action_masks=action_masks)
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


class AgentModel(BaseAgentModel):
    """
    Agent which uses passed model for all moves
    """
    def DoMove(self, game):

        # For JSettlers
        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            #raise Exception("\n\nReturning None Action - INVESTIGATE\n\n")
            return None
        
        possibleActions = self.GetPossibleActions(game.gameState)
        print(f"POSSIBLE_ACTIONS:")
        for action in possibleActions:
            if action:
                print(f"     {action.type}")
        if len(possibleActions) == 1:
            return possibleActions[0]

        actionObj = self.getModelAction(game, possibleActions)

        print(f"SELECTED_ACTION: {actionObj.type}\n")

        if self.playerTrading and actionObj.type == "MakeTradeOffer":
            self.tradeCount += 1

        return actionObj


class AgentMultiModel(BaseAgentModel):
    """
    Agent which uses a separate model for the setup phase and rest of game
    Must pass flag for whether setup is just settlements or settlements and roads
    If 'model' not passed will use random actions
    """

    def __init__(self, name, seatNumber, setupModel: MaskablePPO, fullSetup: bool, playerTrading: bool=False, model: MaskablePPO = None, recordStats=False):

        super(AgentMultiModel, self).__init__(name, seatNumber, model, playerTrading, recordStats=recordStats)
        self.setupModel = setupModel
        self.fullSetup = fullSetup

    def DoMove(self, game):

        # Needed for JSettlers
        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            #raise Exception("\n\nReturning None Action - INVESTIGATE\n\n")
            return None

        possibleActions = self.GetPossibleActions(game.gameState)
        if len(possibleActions) == 1:
            return possibleActions[0]
        
        if game.gameState.currState == "START1A" or game.gameState.currState == "START2A":
            return self.getSetupModelAction(game, possibleActions)
        elif game.gameState.currState == "START1B" or game.gameState.currState == "START2B":
            if self.fullSetup:
                return self.getSetupModelAction(game, possibleActions)
            else:
                return self.getRandomAction(game, possibleActions)

        if self.model == None:
            return self.getRandomAction(game, possibleActions)
        else:
            return self.getModelAction(game, possibleActions)
    
    def getSetupModelAction(self, game, possibleActions):
        """
        Uses setupModel and setupEnv to get action
        """
        action_masks, indexActionDict = self.setupModel.getActionMask(possibleActions)
        state = self.setupModel.getObservation(game.gameState, self.seatNumber)
        action, _states = self.setupModel.predict(state, action_masks=action_masks)
        actionObj = indexActionDict[action.item()]
        return actionObj