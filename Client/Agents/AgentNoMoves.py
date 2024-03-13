from Game.CatanGame import GameState
from Game.CatanPlayer import Player
from Game.CatanAction import *
from Agents.AgentRandom2 import AgentRandom2
import random
from DeepLearning.PPO import MaskablePPO


class AgentNoMoves(AgentRandom2):
    """
    Agent which will choose random setup and then will roll dice and endturn
    """

    def DoMove(self, game):

        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            #raise Exception("\n\nReturning None Action - INVESTIGATE\n\n")
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        if len(possibleActions) == 1:
            return possibleActions[0]

        if game.gameState.currState == "PLAY" or game.gameState.currState == "PLAY1":
            for action in possibleActions:
                if action.type == 'EndTurn' or action.type == 'RollDices':
                    return action
        
        randIndex = random.randint(0, len(possibleActions)-1)
        chosenAction = possibleActions[randIndex]
        
        if chosenAction.type == "MakeTradeOffer":
            self.tradeCount += 1
        
        return chosenAction
        # NOTE: If no actions returned should we return EndTurn/RollDice?
