from Game.CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanPlayer import Player
from Game.CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from Game.CatanAction import *
import numpy as np

def getSetupObservation(gameState: GameState):
    """
    returns dotTotal for each node (54)
    """

    # Get Node info
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getSetupNodeRepresentation(nodes[nodeIndex], gameState))
    
    return np.array(nodeInfo)

setupRandomLowerBounds = np.array(54 * (22*[0]))
var = ([1] * 16) + (5 * [5]) + [15]
setupRandomUpperBounds = np.array(54 * var)
def getSetupRandomObservation(gameState: GameState):
    """
    reutrns full Node representation for each node (54*22)
    """

    # Get Node info
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentation(nodes[nodeIndex], gameState))
    
    return np.array(nodeInfo)

setupRandomWithRoadsLowerBounds = np.array((54 * (22*[0])) + (72 * (5*[0])))
var = ([1] * 16) + (5 * [5]) + [15]
setupRandomWithRoadsUpperBounds = np.array((54 * var) + (72 * (5*[1])))
def getSetupRandomWithRoadsObservation(gameState: GameState):
    """
    Returns all node and edge info
    """

    # Get Node info
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentation(nodes[nodeIndex], gameState))

    edgeInfo = []
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState))
    
    output = [*nodeInfo, *edgeInfo]
    return np.array(output)

# not including recieving trading phase
# [startSet, startRoad, Play, play1, PlacingRobber, Waitingfordiscards, waitingForChoice, placingfreeroad]
phaseOneHotMapping = {
    "START1A":              [1, 0, 0, 0, 0, 0, 0, 0],
    "START1B":              [0, 1, 0, 0, 0, 0, 0, 0],
    "START2A":              [1, 0, 0, 0, 0, 0, 0, 0],
    "START2B":              [0, 1, 0, 0, 0, 0, 0, 0],
    "PLAY":                 [0, 0, 1, 0, 0, 0, 0, 0],
    "PLAY1":                [0, 0, 0, 1, 0, 0, 0, 0],
    "PLACING_ROBBER":       [0, 0, 0, 0, 1, 0, 0, 0],
    "WAITING_FOR_DISCARDS": [0, 0, 0, 0, 0, 1, 0, 0],
    "WAITING_FOR_CHOICE":   [0, 0, 0, 0, 0, 0, 1, 0],
    "PLACING_FREE_ROAD1":   [0, 0, 0, 0, 0, 0, 0, 1],
    "PLACING_FREE_ROAD2":   [0, 0, 0, 0, 0, 0, 0, 1],
}

# returns 2350 length 1D array (now 2358)
def getObservation(gameState: GameState):
    """
    Returns all game info for a model (2358)
    """

    player:Player = gameState.players[0]
    # TODO: if its in same order everytime just fetch
    player1 = gameState.players[1]
    player2 = gameState.players[2]
    player3 = gameState.players[3]

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
    # nodes = gameState.boardNodes
    # nodeInfo = []
    # for nodeIndex in constructableNodesList:
    #     nodeInfo.extend(getNodeRepresentation(nodes[nodeIndex], gameState))

    # Get hex info
    hexes = gameState.boardHexes
    hexInfo = []
    for hexIndex in constructableHexesList:
        hexInfo.extend(getHexRepresentation(hexes[hexIndex], gameState))

    # Get edge info
    edgeInfo = []
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState))

    # Get Game phase
    phase = phaseOneHotMapping[gameState.currState]

    output = [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, roadCount, *longestRoadPlayer, *largestArmyPlayer, player1VP, player2VP, player3VP, *hexInfo, *edgeInfo, *phase]

    return np.array(output)

# returns 2350 length 1D array (now 2358)
def getObservationNoPhase(gameState: GameState):
    """
    Returns all game info for a model, excluding the game phase (2350)
    """

    player:Player = gameState.players[0]
    # TODO: if its in same order everytime just fetch
    player1 = gameState.players[1]
    player2 = gameState.players[2]
    player3 = gameState.players[3]

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
    # nodes = gameState.boardNodes
    # nodeInfo = []
    # for nodeIndex in constructableNodesList:
    #     nodeInfo.extend(getNodeRepresentation(nodes[nodeIndex], gameState))

    # Get hex info
    hexes = gameState.boardHexes
    hexInfo = []
    for hexIndex in constructableHexesList:
        hexInfo.extend(getHexRepresentation(hexes[hexIndex], gameState))

    # Get edge info
    edgeInfo = []
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState))

    # Get Game phase
    phase = phaseOneHotMapping[gameState.currState]

    output = [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, roadCount, *longestRoadPlayer, *largestArmyPlayer, player1VP, player2VP, player3VP, *hexInfo, *edgeInfo]

    return np.array(output)



def getVisibleVictoryPoints(player: Player) -> int:
    """
    Get players victory points not including dev card vp's
    """

    constructionPoints = len(player.settlements) + len(player.cities) * 2

    achievementPoints = 0
    if player.biggestRoad:
        achievementPoints += 2
    if player.biggestArmy:
        achievementPoints += 2

    return constructionPoints + achievementPoints



def getSetupNodeRepresentation(node: BoardNode, gameState: GameState) -> list:
    """
    For each node get: owner, constructionType, portType, dotList, production
    """

    # Get production of a given node
    dotList = [0, 0, 0, 0, 0]
    adjTileNumbers = node.GetAdjacentHexes()
    adjTiles = [gameState.boardHexes[tileNumber] for tileNumber in adjTileNumbers if tileNumber != None]
    for tile in adjTiles:
        if tile.production == None:
            continue
        dotList[resourceIndex[tile.production]] += numberDotsMapping[tile.number]
    dotTotal = sum(dotList)

    #       cat    cat               cat       num          num
    return [dotTotal]



def getNodeRepresentation(node: BoardNode, gameState: GameState) -> list:
    """
    For each node get: owner, constructionType, portType, dotList, production (22 total)
    """
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

    #       cat    cat               cat       num          num
    return [*owner, *constructionType, *portType, *dotList, dotTotal]



def getHexRepresentation(hex: BoardHex, gameState: GameState) -> list:
    """
    For each hex get number, resource, for each surrounding node get: owner, construction type,
    """
    dot = numberDotsMapping[hex.number]
    resource = [0, 0, 0, 0, 0, 0]
    resource[resourceIndex[hex.production]] = 1
    
    # For each adjacent node add owner, construction type
    adjNodesInfo = []
    nodeIndexes = hex.GetAdjacentNodes()
    for nodeIndex in nodeIndexes:
        owner = [0, 0, 0, 0, 0]
        constructionType = [0, 0, 0, 0]
        portType = [0, 0, 0, 0, 0, 0, 0]
        node = gameState.boardNodes[nodeIndex]
        portType[portTypeIndex[node.portType]] = 1
        if node.construction == None:
            owner[-1] = 1
            constructionType[-1] = 1
        else:
            owner[node.construction.owner] = 1
            constructionType[constructionTypeIndex[node.construction.type]] = 1
        adjNodesInfo.extend([*owner, *constructionType, *portType])
    #       num,   cat,      (cat, cat)
    return [dot, *resource, *adjNodesInfo]



def getEdgeRepresentation(edge: BoardEdge, gameState: GameState) -> list:
    """
    For each edge get owner
    """
    owner = [0, 0, 0, 0, 0]
    if edge.construction == None:
        owner[-1] = 1
    else:
        # cat
        owner[edge.construction.owner] = 1
    return owner



myResourcesLower = [0] * 6
myResourcesUpper = [10] * 6 
developmentCardsLower = [0] * 5 
developmentCardsUpper = [10] * 5
myVictoryPointsLower = [0]
myVictoryPointsUpper = [10]
moreThan7ResourcesLower = [0]
moreThan7ResourcesUpper = [1]
tradeRatesLower = [2] * 5
tradeRatesUpper = [4] * 5
knightsLower = [0]
knightsUpper = [10]
roadCountLower = [0]
roadCountUpper = [10]
longestRoadPlayerLower = [0] * 5
longestRoadPlayerUpper = [1] * 5
largestArmyPlayerLower = [0] * 5
largestArmyPlayerUpper = [1] * 5
player1VP_lower = [0]
player1VP_upper = [1]
player2VP_lower = [0]
player2VP_upper = [1]
player3VP_lower = [0]
player3VP_upper = [1]
# 19 hexes
# for each hex: dot - 0->15, resource(6) 0->1, adjNodeInfo
# for each 6 adjNode(16): owner(5)0->1, constructType(4)0->1, portType(7)0->1
hexInfoLowerList = 19 * ((7 + 6*16) * [0])
hexInfoUpper = [15] + ((6 + 6*16) * [0])
hexInfoUpperList = 19 * hexInfoUpper
edgeInfoLowerList = 72 * [0, 0, 0, 0, 0]
edgeInfoUpperList = 72 * [1, 1, 1, 1, 1]
phaseLower = [0] * 8
phaseUpper = [1] * 8

lowerBounds = np.array([
    *myResourcesLower,
    *developmentCardsLower,
    *myVictoryPointsLower,
    *moreThan7ResourcesLower,
    *tradeRatesLower,
    *knightsLower,
    *roadCountLower,
    *longestRoadPlayerLower,
    *largestArmyPlayerLower,
    *player1VP_lower,
    *player2VP_lower,
    *player3VP_lower,
    *hexInfoLowerList,
    *edgeInfoLowerList,
    *phaseLower
])

upperBounds = np.array([
    *myResourcesUpper,
    *developmentCardsUpper,
    *myVictoryPointsUpper,
    *moreThan7ResourcesUpper,
    *tradeRatesUpper,
    *knightsUpper,
    *roadCountUpper,
    *longestRoadPlayerUpper,
    *largestArmyPlayerUpper,
    *player1VP_upper,
    *player2VP_upper,
    *player3VP_upper,
    *hexInfoUpperList,
    *edgeInfoUpperList,
    *phaseUpper
])

# Other possible inputs
    # possibleRoads = player.possibleRoads
    # possibleSettlements = player.possibleSettlements



# 100: 7.65s
# 100: getNodeRepresentation, 4.8 = 2.8
# 100: getHexRepresentation, 6.3 = 1.3
# 100: getEdgeRepresntation, 7.1 = 0.5
# 100: the rest - 5.2 = 2.5
