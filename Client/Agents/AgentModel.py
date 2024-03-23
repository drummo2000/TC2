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
    def __init__(self, name, seatNumber, model: MaskablePPO, playerTrading: bool=False, recordStats=False, jsettlersGame=False):

        super(BaseAgentModel, self).__init__(name, seatNumber, playerTrading=playerTrading, recordStats=recordStats, jsettlersGame=jsettlersGame)
        self.model                  = model
        self.jsettlersGame = jsettlersGame

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

        if self.jsettlersGame:
            print(f"POSSIBLE_ACTIONS: {self.resources[:5]}, DevCards: {self.developmentCards}")
            for action in possibleActions:
                if action:
                    print(f"     {action.type}")

        if len(possibleActions) == 1:
            return possibleActions[0]
        
        # Don't Allow to build roads when theres a possible settlement and we haven't built our 1st settlement
        if self.jsettlersGame:
            if game.gameState.currState[:5] != "START":
                canBuildRoad = False
                canBuyDevCard = False
                for action in possibleActions:
                    if action.type == "BuildRoad":
                        canBuildRoad = True
                    elif action.type == "BuyDevelopmentCard":
                        canBuyDevCard = True
                # Remove road option if haven't built 1st settlement or have longest road
                if canBuildRoad:
                    if ((len(self.settlements) + len(self.cities) <= 2) and len(game.gameState.GetPossibleSettlements(self)) > 0) or (game.gameState.longestRoadPlayer == self.seatNumber):
                        possibleActions = [action for action in possibleActions if action.type != "BuildRoad"]
                        print("                 REMOVED BUILD ROAD OPTIONS")
                # Remove buy dev card option if we haven't built a city
                if canBuyDevCard:
                    if len(self.cities) == 0:
                        possibleActions = [action for action in possibleActions if action.type != "BuyDevelopmentCard"]
                        print("                 REMOVED BUY DEVCARD OPTIONS")
                


        actionObj = self.getModelAction(game, possibleActions)

        if self.jsettlersGame:
            if actionObj.type == "MakeTradeOffer":
                print(f"SELECTED_ACTION: {actionObj.type}, {actionObj.giveResources[:5]}_{actionObj.getResources[:5]}\n")
            else:
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

    def __init__(self, name, seatNumber, setupModel: MaskablePPO, fullSetup: bool, playerTrading: bool=False, model: MaskablePPO = None, recordStats=False, jsettlersGame=False):

        super(AgentMultiModel, self).__init__(name, seatNumber, model, playerTrading, recordStats=recordStats, jsettlersGame=jsettlersGame)
        self.setupModel = setupModel
        self.fullSetup = fullSetup

    def DoMove(self, game):

        # Needed for JSettlers
        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            #raise Exception("\n\nReturning None Action - INVESTIGATE\n\n")
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        if self.jsettlersGame:
            print(f"POSSIBLE_ACTIONS: {self.resources[:5]}, DevCards: {self.developmentCards}")
            for action in possibleActions:
                if action:
                    print(f"     {action.type}")

        if len(possibleActions) == 1:
            return possibleActions[0]
        
        if game.gameState.currState == "START1A" or game.gameState.currState == "START2A":
            actionObj = self.getSetupModelAction(game, possibleActions)
        elif game.gameState.currState == "START1B" or game.gameState.currState == "START2B":
            if self.fullSetup:
                actionObj = self.getSetupModelAction(game, possibleActions)
            else:
                actionObj = self.getRandomAction(game, possibleActions)
        
        if self.jsettlersGame:
            # Don't Allow to build roads when theres a possible settlement and we haven't built our 1st settlement
            if game.gameState.currState[:5] != "START":
                canBuildRoad = False
                for action in possibleActions:
                    if action.type == "BuildRoad":
                        canBuildRoad = True
                        break
                if canBuildRoad:
                    if (len(self.settlements) + len(self.cities) <= 2) and len(game.gameState.GetPossibleSettlements(self)) > 0:
                        possibleActions = [action for action in possibleActions if action.type != "BuildRoad"]
                        print("                 REMOVED BUILD ROAD OPTIONS")

        if self.model == None:
            actionObj = self.getRandomAction(game, possibleActions)
        else:
            actionObj = self.getModelAction(game, possibleActions)
        
        if self.jsettlersGame:
            if actionObj.type == "MakeTradeOffer":
                print(f"SELECTED_ACTION: {actionObj.type}, {actionObj.giveResources[:5]}_{actionObj.getResources[:5]}\n")
            else:
                print(f"SELECTED_ACTION: {actionObj.type}\n")

        if self.playerTrading and actionObj.type == "MakeTradeOffer":
            self.tradeCount += 1

        return actionObj
    
    def getSetupModelAction(self, game, possibleActions):
        """
        Uses setupModel and setupEnv to get action
        """
        action_masks, indexActionDict = self.setupModel.getActionMask(possibleActions)
        state = self.setupModel.getObservation(game.gameState, self.seatNumber)
        action, _states = self.setupModel.predict(state, action_masks=action_masks)
        actionObj = indexActionDict[action.item()]
        return actionObj