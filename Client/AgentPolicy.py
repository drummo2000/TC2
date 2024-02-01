from CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from CatanPlayer import Player
from CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from CatanAction import *
from itertools import combinations
import math
import random
from ActionMask import getActionMask
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ActionMask import allActionsList
from AgentRandom2 import AgentRandom2
from ModelState import getInputState
from PPO import PPO

class AgentPolicy(AgentRandom2):

    def __init__(self, name, seatNumber, network: PPO, playerTrading=False):

        super(AgentPolicy, self).__init__(name, seatNumber, playerTrading)
        self.agentName              = name
        self.network                = network
        self.doRandomMoves          = doRandomMoves
    
    # Return selected action
    def DoMove(self, game) -> Action:

        # If not my turn and were not in WAITING_FOR_DISCARDS phase then return None
        if game.gameState.currPlayer != self.seatNumber and \
            game.gameState.currState != "WAITING_FOR_DISCARDS":
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        if type(possibleActions) != list:
            possibleActions = [possibleActions]

        if len(possibleActions) == 1:
            return possibleActions[0]
        else:
            actionMask, indexActionDict = getActionMask(possibleActions)
            state = getInputState(game.gameState)

            actionIndex = self.network.select_action(state, actionMask)

            self.network.buffer.rewards.append(0)
            self.network.buffer.is_terminals.append(False)

            return indexActionDict[actionIndex]
