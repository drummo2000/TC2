from Game.CatanGame import GameState
from Game.CatanPlayer import Player
from Game.CatanAction import *
from Agents.AgentRandom2 import AgentRandom2
import random
import torch.nn as nn
from DeepLearning.GetObservation import getObservationFull
from DeepLearning.GetActionMask import getActionMask
import torch


class BaseCopyUCT(AgentRandom2):
    """
    The Agents here are not used for training, they are either used as opponents for training
    or for testing pretrained models
    """
    def __init__(self, name, seatNumber, model: nn.Module, playerTrading: bool=False):

        super(BaseCopyUCT, self).__init__(name, seatNumber, playerTrading)
        self.model                  = model

    def getModelAction(self, game, possibleActions):
        """
        Uses model and env to get action
        """
        state = getObservationFull(game.gameState)
        action_masks, indexActionDict = getActionMask(possibleActions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_index = self.model(state)
        try:
            actionObj = indexActionDict[action_index]
        except Exception as e:
            print(f"\nWRONG ACTION SELECTED, SELECTING RANDOM ACTION")
            return self.getRandomAction(possibleActions)
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




class AgentCopyUCT(BaseCopyUCT):
    """
    Agent which uses passed model for all moves
    """
    def DoMove(self, game):
        possibleActions = self.GetPossibleActions(game.gameState)
        if len(possibleActions) == 1:
            return possibleActions[0]

        return self.getModelAction(game, possibleActions)

    


class AgentMultiCopyUCT(BaseCopyUCT):
    """
    Agent which uses a separate model for the setup phase and rest of game
    Must pass flag for whether setup is just settlements or settlements and roads
    If 'model' not passed will use random actions
    """

    def __init__(self, name, seatNumber, setupModel: nn.Module, fullSetup: bool, playerTrading: bool=False, model: nn.Module = None):

        super(AgentMultiCopyUCT, self).__init__(name, seatNumber, model, playerTrading)
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
        state = getObservationFull(game.gameState)
        action_masks, indexActionDict = getActionMask(possibleActions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_logits = self.setupModel(state)
            tensor_mask = torch.tensor(action_masks[:126], dtype=torch.float)
            masked_logits = action_logits.masked_fill(tensor_mask == 0, float('-inf'))

            action_index = torch.argmax(masked_logits)
        try:
            actionObj = indexActionDict[action_index.item()]
        except Exception as e:
            print(f"\nWRONG ACTION SELECTED, SELECTING RANDOM ACTION: {e}")
            raise e
        return actionObj