from CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from CatanPlayer import Player
from CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from CatanAction import *
import numpy as np

def getSetupInputState(gameState: GameState):
    player:Player = next(filter(lambda p: p.name=="P0", gameState.players), None)

    # Get Node info
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getSetupNodeRepresentation(nodes[nodeIndex], gameState))
    
    return np.array(nodeInfo)


# returns 2350 length 1D array
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

    output = [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, roadCount, *longestRoadPlayer, *largestArmyPlayer, player1VP, player2VP, player3VP, *hexInfo, *edgeInfo]

    return np.array(output)

# Get players victory points not including dev card vp's
def getVisibleVictoryPoints(player: Player) -> int:

    constructionPoints = len(player.settlements) + len(player.cities) * 2

    achievementPoints = 0
    if player.biggestRoad:
        achievementPoints += 2
    if player.biggestArmy:
        achievementPoints += 2

    return constructionPoints + achievementPoints

# For each node get: owner, constructionType, portType, dotList, production
def getSetupNodeRepresentation(node: BoardNode, gameState: GameState) -> list:

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

    #       cat    cat               cat       num          num
    return [*owner, *constructionType, *portType, *dotList, dotTotal]

# For each hex get number, resource, for each surrounding node get: owner, construction type,
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



# 100: 7.65s
# 100: getNodeRepresentation, 4.8 = 2.8
# 100: getHexRepresentation, 6.3 = 1.3
# 100: getEdgeRepresntation, 7.1 = 0.5
# 100: the rest - 5.2 = 2.5
