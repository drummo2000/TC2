from Game.CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanPlayer import Player
from Game.CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from Game.CatanAction import *
from DeepLearning.GetObservation import getVisibleVictoryPoints, phaseOneHotMapping
import numpy as np

phaseOneHotMappingSetup = {
    "START1A":              [1, 0],
    "START1B":              [0, 1],
    "START2A":              [1, 0],
    "START2B":              [0, 1]
}

otherInfoLowerBound = 13*[0]
otherInfoUpperBound = 5*[4] + 5*[40] + 2*[1] + [150]

# 13 total
def getOtherInfo(gameState, playerNumber):
    player:Player = gameState.players[playerNumber]

    ## My info ##
    tradeRates = player.tradeRates # 5
    
    # Current Production
    resourceProduction = listm([0, 0, 0, 0, 0, 0]) # 5
    for diceNumber, resourceList in player.diceProduction.items():
        resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
    resourceProduction = resourceProduction[:-1]

    # Get Game phase
    phase = phaseOneHotMappingSetup[gameState.currState] # 2

    playerTurns = player.playerTurns # 1

    return [*tradeRates, *resourceProduction, *phase, playerTurns]


hexLowerBound = [0]*9
hexUpperBound = [5] + 8*[1]
hexsLowerBound = hexLowerBound * 19
hexsUpperBound = hexUpperBound * 19
def getHexRepresentation(hex: BoardHex, gameState: GameState, playerNumber=0) -> list:
    """
    For each hex get number, resource, for each surrounding node get: owner, construction type, 19 * 9 = 171
    """
    dot = numberDotsMapping[hex.number]
    underOver7 = [0, 0]
    if hex.number < 7:
        underOver7[0] = 1
    else:
        underOver7[1] = 1
    resource = [0, 0, 0, 0, 0, 0]
    resource[resourceIndex[hex.production]] = 1

    #        1      2             6
    return [dot, *underOver7, *resource]


nodeLowerBound = 10*[0]
nodeUpperBound = 10*[1]
nodesLowerBound = nodeLowerBound*54
nodesUpperBound = nodeUpperBound*54
def getNodeRepresentation(node: BoardNode, gameState: GameState, playerNumber=0) -> list:
    """
    For each node get: 54*10 = 540
    """
    owner = [0, 0, 0] # me, other, nobody
    portType = [0, 0, 0, 0, 0, 0, 0]

    if node.construction != None:
        owner = [1, 0, 0] if node.construction.owner == playerNumber else [0, 1, 0]
    else:
        owner[-1] = 1

    portType[portTypeIndex[node.portType]] = 1

    #       3           7            
    return [*owner, *portType]


edgeLowerBound = [0]*3
edgeUpperBound = [1]*3
edgesLowerBound = 72*edgeLowerBound
edgesUpperBound = 72*edgeUpperBound
def getEdgeRepresentation(edge: BoardEdge, gameState: GameState, playerNumber=0) -> list:
    """
    For each edge get owner, 3*72=216
    """
    owner = [0, 0, 0]
    if edge.construction == None:
        owner[-1] = 1
    else:
        # cat
        owner = [1, 0, 0] if edge.construction.owner == playerNumber else [0, 1, 0]

    #        3
    return owner


lowerBound = np.array([*otherInfoLowerBound, *hexsLowerBound, *nodesLowerBound, *edgesLowerBound])
upperBound = np.array([*otherInfoUpperBound, *hexsLowerBound, *nodesUpperBound, *edgesUpperBound])
# 940
def getObservationSetup(gameState: GameState, playerNumber=0):
    """
    Final observation for thesis after testing multiple variations
    """
    # Other info
    otherInfo = getOtherInfo(gameState, playerNumber)
    
    # Get hex info
    hexes = gameState.boardHexes
    hexInfo = []
    for hexIndex in constructableHexesList:
        hexInfo.extend(getHexRepresentation(hexes[hexIndex], gameState, playerNumber=playerNumber))
    
    # Get node info containing hexes
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentation(nodes[nodeIndex], gameState, playerNumber=playerNumber))
    
    # Get edge info
    edgeInfo = []
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState, playerNumber=playerNumber))

    output = [*otherInfo, *hexInfo, *nodeInfo, *edgeInfo]
    return np.array(output)