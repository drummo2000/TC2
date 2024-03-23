from Game.CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanPlayer import Player
from Game.CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from Game.CatanAction import *
import numpy as np
from DeepLearning.GetObservation import getVisibleVictoryPoints, getHexRobberRating, phaseOneHotMappingFull


def getObservationFull(gameState: GameState):
    """
    Observation used for model to copy MCTS agent actions
    """
    player:Player = gameState.players[0]
    player1 = gameState.players[1]
    player2 = gameState.players[2]
    player3 = gameState.players[3]

    ## My info ##
    myResources = player.resources # 6
    developmentCards = player.developmentCards # 5
    myVictoryPoints = player.victoryPoints # 1
    moreThan7Resources = int(len(myResources) > 7) # 1
    tradeRates = player.tradeRates # 5
    # Number of used knights
    knights = player.knights # 1
    roadCount = player.roadCount # 1
    # Enough resourcs to build settlement/city/road
    canAffordSettlement = int(player.CanAfford(BuildSettlementAction.cost)) # 1
    canAffordCity = int(player.CanAfford(BuildCityAction.cost)) # 1
    canAffordRoad = int(player.CanAfford(BuildRoadAction.cost)) # 1

    ## Other players info ##
    longestRoadPlayer = [0, 0, 0, 0, 0]
    longestRoadPlayer[gameState.longestRoadPlayer] = 1
    largestArmyPlayer = [0, 0, 0, 0, 0]
    largestArmyPlayer[gameState.largestArmyPlayer] = 1
    player1VP = getVisibleVictoryPoints(player1)
    player2VP = getVisibleVictoryPoints(player2)
    player3VP = getVisibleVictoryPoints(player3)

    ## Board info ##

    # Get hex robber info
    hexes = gameState.boardHexes
    hexInfo = [] # 19
    for hexIndex in constructableHexesList:
        hexInfo.append(getHexRobberRating(hexes[hexIndex], gameState))
    
    # Get node info
    nodes = gameState.boardNodes
    nodeInfo = [] # 54 * 20 = 1080
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentationFull(nodes[nodeIndex], gameState))

    # Get edge info
    edgeInfo = [] # 72 * 4 = 288
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentationFull(gameState.boardEdges[edgeIndex], gameState))

    # Get Game phase
    phase = phaseOneHotMappingFull[gameState.currState] # 8

    output = [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, roadCount, canAffordSettlement, canAffordCity, canAffordRoad, *longestRoadPlayer, *largestArmyPlayer, player1VP, player2VP, player3VP, *nodeInfo, *hexInfo, *edgeInfo, *phase]
    return np.array(output)


def getNodeRepresentationFull(node: BoardNode, gameState: GameState) -> list:
    """
    For each node get: owner, constructionType, portType, dotList, production (20 total) - 20*54 = 1080
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

    setupPhase = not gameState.setupDone
    canBuildSettlement = int(gameState.CanBuildSettlement(gameState.players[0], node, setUpPhase=setupPhase))

    #       cat    cat               cat       num          num
    return [*owner, *constructionType, *portType, *dotList, dotTotal, canBuildSettlement]



def getEdgeRepresentationFull(edge: BoardEdge, gameState: GameState) -> list:
    """
    For each edge get owner - 4 * 72 = 288
    """
    owner = [0, 0, 0, 0, 0]
    if edge.construction == None:
        owner[-1] = 1
    else:
        # cat
        owner[edge.construction.owner] = 1

    setupPhase = not gameState.setupDone
    canBuildRoad = int(gameState.CanBuildRoad(gameState.players[0], edge, edge.index, setUpPhase=setupPhase))

    return [*owner, canBuildRoad]