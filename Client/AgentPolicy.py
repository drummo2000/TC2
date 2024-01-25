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

class PolicyNetwork(nn.Module):
    # sets up the shape of the NN and the optimizer used to update parameters
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # Takes in the current state and returns the probabilities of taking the action
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state, action_mask):
        state = torch.from_numpy(state).float().reshape(-1).unsqueeze(0)

        probs = self.forward(state)
        # Must reduce prob of masked actions to 0
        probs = probs * torch.from_numpy(action_mask).float().reshape(-1).unsqueeze(0)

        # Normalize the masked probabilities
        sum_masked_probs = torch.sum(probs)
        probs = probs / sum_masked_probs

        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


class AgentPolicy(AgentRandom2):

    def __init__(self, name, seatNumber, network):

        super(AgentPolicy, self).__init__(name, seatNumber)
        self.agentName              = name
        self.network                = network
    
    # Return selected action
    def DoMove(self, game):

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

            # action is the index of selected action
            actionIndex, log_prob = self.network.get_action(state, actionMask)
            return indexActionDict[actionIndex]


        # NOTE: If no actions returned should we return EndTurn/RollDice?




####################################################################################################################################
####################################################################################################################################
####################################################################################################################################




# Need to return 1D array with all info for policy
def getInputState(gameState: GameState):
    player:Player = next(filter(lambda p: p.name=="P0", gameState.players), None)
    # TODO: if its in same order everytime just fetch
    player1 = next(filter(lambda p: p.name=="P1", gameState.players), None)
    player2 = next(filter(lambda p: p.name=="P2", gameState.players), None)
    player3 = next(filter(lambda p: p.name=="P3", gameState.players), None)

    ## My info ##
    myResources = player.resources
    developmentCards = player.developmentCards
    myVictoryPoints = player.victoryPoints
    moreThan7Resources = int(len(myResources) > 7)
    tradeRates = player.tradeRates
    knights = player.knights
    roadCount = player.roadCount

    ## Other players info ##
    longestRoadPlayer = [0, 0, 0, 0, 0]
    longestRoadPlayer[gameState.longestRoadPlayer] = 1
    largestArmyPlayer = [0, 0, 0, 0, 0]
    largestArmyPlayer[gameState.largestArmyPlayer] = 1
    player1VP = getVisibleVictoryPoints(player1)
    player2VP = getVisibleVictoryPoints(player2)
    player3VP = getVisibleVictoryPoints(player3)

    # Get Node info
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentation(nodes[nodeIndex], gameState))

    # Get hex info
    hexes = gameState.boardHexes
    hexInfo = []
    for hexIndex in constructableHexesList:
        hexInfo.extend(getHexRepresentation(hexes[hexIndex], gameState))

    # Get edge info
    edgeInfo = []
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState))

    output = [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, roadCount, *longestRoadPlayer, *largestArmyPlayer, player1VP, player2VP, player3VP, *nodeInfo, *hexInfo, *edgeInfo]
    return np.array(output)

# Get players victory points not including dev card vp's
def getVisibleVictoryPoints(player: Player) -> int:

    constructionPoints = len(player.settlements) + len(player.cities) * 2

    achievementPoints = 0
    if player.biggestRoad:
        achievementPoints += 2
    if player.biggestArmy:
        achievementPoints += 2

    player.victoryPoints = constructionPoints + achievementPoints

    return player.victoryPoints


# For each node get: owner, constructionType, portType, dotList, production
def getNodeRepresentation(node: BoardNode, gameState: GameState) -> list:
    owner = [0, 0, 0, 0, 0]
    constructionType = [0, 0, 0, 0]
    portType = [0, 0, 0, 0, 0, 0, 0]

    if node.construction != None:
        owner[node.construction.owner] = 1
        constructionType[constructionTypeIndex[node.construction.type]] = 1
    else:
        owner[-1] = 1
        constructionType[-1] = 1

    portType[portTypeIndex[node.portType]] = 1

    # Get production of a given node
    dotList = [0, 0, 0, 0, 0]
    adjTileNumbers = node.GetAdjacentHexes()
    adjTiles = [gameState.boardHexes[tileNumber] for tileNumber in adjTileNumbers if tileNumber != None]
    for tile in adjTiles:
        if tile.production == None:
            continue
        dotList[resourceIndex[tile.production]] += numberDotsMapping[tile.number]

    dotTotal = sum(dotList)
    #       cat    cat               cat       num             num
    return [*owner, *constructionType, *portType, *dotList, dotTotal]

# For each hex get number, resource, for each surrounding node get: owner, resource type,
def getHexRepresentation(hex: BoardHex, gameState: GameState) -> list:
    dot = numberDotsMapping[hex.number]
    resource = [0, 0, 0, 0, 0, 0]
    resource[resourceIndex[hex.production]] = 1
    
    # For each adjacent node add owner, construction type
    adjNodesInfo = []
    nodeIndexes = hex.GetAdjacentNodes()
    for nodeIndex in nodeIndexes:
        owner = [0, 0, 0, 0, 0]
        constructionType = [0, 0, 0, 0]
        node = gameState.boardNodes[nodeIndex]
        if node.construction == None:
            owner[-1] = 1
            constructionType[-1] = 1
        else:
            owner[node.construction.owner] = 1
            constructionType[constructionTypeIndex[node.construction.type]] = 1
        adjNodesInfo.extend([*owner, *constructionType])
    #       num,   cat,      (cat, cat)
    return [dot, *resource, *adjNodesInfo]

# For each edge get owner
def getEdgeRepresentation(edge: BoardEdge, gameState: GameState) -> list:
    owner = [0, 0, 0, 0, 0]
    if edge.construction == None:
        owner[-1] = 1
    else:
        # cat
        owner[edge.construction.owner] = 1
    return owner





# Other possible inputs
    # possibleRoads = player.possibleRoads
    # possibleSettlements = player.possibleSettlements