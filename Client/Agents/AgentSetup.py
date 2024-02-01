from CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from CatanPlayer import Player
from CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from CatanAction import *
from itertools import combinations
import math
import random
from ActionMask import getActionMask, getSetupActionMask
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ActionMask import allActionsList
from .AgentRandom2 import AgentRandom2
from ModelState import getInputState, getSetupInputState
from PPO import PPO

"""
Agent which uses model for placement phase and random actions for rest of game
"""
class AgentSetup(AgentRandom2):

    def __init__(self, name, seatNumber, network: PPO, playerTrading=False):

        super(AgentSetup, self).__init__(name, seatNumber, playerTrading)
        self.agentName              = name
        self.network                = network

    
    # Return selected action
    def DoMove(self, game) -> Action:

        if game.gameState.currState == "START1A" or game.gameState.currState == "START2A":

            possibleActions = self.GetAllPossibleActions_Setup(game.gameState, self)

            actionMask, indexActionDict = getSetupActionMask(possibleActions)
            state = getSetupInputState(game.gameState)

            actionIndex = self.network.select_action(state, actionMask)

            self.network.buffer.rewards.append(0)
            self.network.buffer.is_terminals.append(False)

            return indexActionDict[actionIndex]
        # Otherwise use random moves
        else:
            # If not my turn and were not in WAITING_FOR_DISCARDS phase then return None
            if game.gameState.currPlayer != self.seatNumber and \
                game.gameState.currState != "WAITING_FOR_DISCARDS":
                return None

            possibleActions = self.GetPossibleActions(game.gameState)

            if type(possibleActions) != list:
                # possibleActions = [possibleActions]
                return possibleActions
            
            randIndex = random.randint(0, len(possibleActions)-1)
            chosenAction = possibleActions[randIndex]
            
            if chosenAction.type == "MakeTradeOffer":
                self.tradeCount += 1
            
            return chosenAction
            # NOTE: If no actions returned should we return EndTurn/RollDice?