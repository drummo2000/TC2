from Game.CatanGame import GameState
from Game.CatanPlayer import Player
from Game.CatanAction import *
from Agents.AgentRandom2 import AgentRandom2
import random
from sb3_contrib.ppo_mask import MaskablePPO


class AgentModel(AgentRandom2):
    """
    Agent which uses trained model to decide moves.
    Must pass the environment model was trained on for it to work.
    Can be used for SetupOnly/NoSetup/Combination models
    """

    def __init__(self, name, seatNumber, model: MaskablePPO=None, env=None, setupModel: MaskablePPO=None, setupEnv=None, playerTrading: bool=False):

        super(AgentModel, self).__init__(name, seatNumber)
        self.agentName              = name
        self.playerTrading          = playerTrading
        self.model                  = model
        self.setupModel             = setupModel
        self.setupGetActionMaskFunc = setupEnv.getActionMaskFunc if setupEnv else None
        self.setupGetInputStateFunc = setupEnv.getInputStateFunc if setupEnv else None
        self.setupPhases            = setupEnv.phases if setupEnv else []
        self.getActionMaskFunc      = env.getActionMaskFunc if env else None
        self.getInputStateFunc      = env.getInputStateFunc if env else None


    def DoMove(self, game):
        possibleActions = self.GetPossibleActions(game.gameState)
        if len(possibleActions) == 1:
            return possibleActions[0]

        # Setup phase
        if not game.gameState.setupDone:
            if game.gameState.currState in self.setupPhases:
                return self.getSetupActionObj(game, possibleActions)
            else:
                return self.getRandomAction(game, possibleActions)

        # Setup over
        if self.model:
            return self.getActionObj(game, possibleActions)
        else:
            return self.getRandomAction(game, possibleActions)


    def getActionObj(self, game, possibleActions):
        """
        Uses model and env to get action
        """
        action_masks, indexActionDict = self.getActionMaskFunc(possibleActions)
        state = self.getInputStateFunc(game.gameState)
        action, _states = self.model.predict(state, action_masks=action_masks)
        actionObj = indexActionDict[action.item()]
        return actionObj


    def getSetupActionObj(self, game, possibleActions):
        """
        Uses setupModel and setupEnv to get action
        """
        action_masks, indexActionDict = self.setupGetActionMaskFunc(possibleActions)
        state = self.setupGetInputStateFunc(game.gameState)
        action, _states = self.setupModel.predict(state, action_masks=action_masks)
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