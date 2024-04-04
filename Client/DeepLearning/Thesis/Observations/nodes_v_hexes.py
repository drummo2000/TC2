from Game.CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanPlayer import Player
from Game.CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from Game.CatanAction import *
from DeepLearning.GetObservation import getVisibleVictoryPoints, phaseOneHotMapping
import numpy as np

otherInfoLowerBound = 54*[0]
otherInfoUpperBound = 5*[10] + 5*[10] + [10] + [1] + 5*[4] + [10] + 5*[1] + [15] + 5*[1] + 9*[1] + 5*[40] + 3*[1] + 8*[1]
# Total: 54
def getOtherInfo(gameState, playerNumber):
    player:Player = gameState.players[playerNumber]
    player1 = gameState.players[(playerNumber+1)%4]
    player2 = gameState.players[(playerNumber+2)%4]
    player3 = gameState.players[(playerNumber+3)%4]

    ## My info ##
    myResources = player.resources[:5] # 5
    developmentCards = player.developmentCards # 5
    myVictoryPoints = player.victoryPoints # 1
    moreThan7Resources = int(len(myResources) > 7) # 1
    tradeRates = player.tradeRates # 5
    # Number of used knights
    knights = player.knights # 1
    largestArmyPlayer = [0, 0, 0, 0, 0] # 5
    largestArmyPlayer[gameState.largestArmyPlayer] = 1
    roadCount = player.roadCount # 1
    longestRoadPlayer = [0, 0, 0, 0, 0] # 5
    longestRoadPlayer[gameState.longestRoadPlayer] = 1
    canAffordSettlement = int(player.CanAfford(BuildSettlementAction.cost)) # 1
    canAffordCity = int(player.CanAfford(BuildCityAction.cost)) # 1
    canAffordRoad = int(player.CanAfford(BuildRoadAction.cost)) # 1
    canAffordDevelopmentCard = int(player.CanAfford(BuyDevelopmentCardAction.cost)) # 1
        # Could build if we can afford
    possibleSettlement = 0 # 1
    if gameState.currState[:5] != "START":
        if gameState.GetPossibleSettlements(player):
            possibleSettlement = 1  
    possibleCity = 1 if player.settlements else 0 # 1
    numSettlementsBuilt = len(player.settlements) + len(player.cities) - 2 # excluding setup - 1
    numCitiesBuilt = len(player.cities) # 1
    numRoadsBuilt = len(player.roads) - 2 # excluding setup - 1
    
    # Current Production
    resourceProduction = listm([0, 0, 0, 0, 0, 0]) # 5
    for diceNumber, resourceList in player.diceProduction.items():
        resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
    resourceProduction = resourceProduction[:-1]

    ## Other players info ##
    player1VP = getVisibleVictoryPoints(player1) # 1
    player2VP = getVisibleVictoryPoints(player2) # 1
    player3VP = getVisibleVictoryPoints(player3) # 1

    # Get Game phase
    phase = phaseOneHotMapping[gameState.currState] # 8

    return [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, *largestArmyPlayer, roadCount, *longestRoadPlayer, canAffordSettlement, canAffordCity, canAffordRoad, canAffordDevelopmentCard, possibleSettlement, possibleCity, numSettlementsBuilt, numCitiesBuilt, numRoadsBuilt, *resourceProduction, player1VP, player2VP, player3VP, *phase]



nodeLowerBound = 21*[0]
nodeUpperBound = 16*[1] + 5*[13]
nodesLowerBound = nodeLowerBound*54
nodesUpperBound = nodeUpperBound*54
def getNodeRepresentation(node: BoardNode, gameState: GameState, playerNumber=0) -> list:
    """
    For each node get: owner, constructionType, portType, dotList (21 total), 54*21 = 1134
    """
    owner = [0, 0, 0, 0, 0]
    constructionType = [0, 0, 0, 0]
    portType = [0, 0, 0, 0, 0, 0, 0]

    if node.construction != None:
        owner[(playerNumber+node.construction.owner)%4] = 1
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

    #       5         4                7          5
    return [*owner, *constructionType, *portType, *dotList]



hexLowerBound = [0]*97
hexUpperBound = [5] + 6*[1] + 90*[1]
hexsLowerBound = hexLowerBound * 19
hexsUpperBound = hexUpperBound * 19
def getHexRepresentation(hex: BoardHex, gameState: GameState, playerNumber=0) -> list:
    """
    For each hex get number, resource, for each surrounding node get: owner, construction type, 19 * (1+6+6*(15))=>97 = 1843
    """
    dot = numberDotsMapping[hex.number]
    resource = [0, 0, 0, 0, 0, 0]
    resource[resourceIndex[hex.production]] = 1
    
    # For each adjacent node add owner, construction type, port type
    adjNodesInfo = []
    nodeIndexes = hex.GetAdjacentNodes()
    for nodeIndex in nodeIndexes:
        owner = [0, 0, 0, 0, 0]
        constructionType = [0, 0, 0]
        portType = [0, 0, 0, 0, 0, 0, 0]
        node = gameState.boardNodes[nodeIndex]
        portType[portTypeIndex[node.portType]] = 1
        if node.construction == None:
            owner[-1] = 1
            constructionType[-1] = 1
        else:
            owner[(playerNumber+node.construction.owner)%4] = 1
            constructionType[constructionTypeIndex[node.construction.type]-1] = 1
        adjNodesInfo.extend([*owner, *constructionType, *portType])
    #       num,   cat,      (cat, cat, cat)
    return [dot, *resource, *adjNodesInfo]



edgeLowerBound = [0]*5
edgeUpperBound = [1]*5
edgesLowerBound = 72*edgeLowerBound
edgesUpperBound = 72*edgeUpperBound
def getEdgeRepresentation(edge: BoardEdge, gameState: GameState, playerNumber=0) -> list:
    """
    For each edge get owner, 5*72=360
    """
    owner = [0, 0, 0, 0, 0]
    if edge.construction == None:
        owner[-1] = 1
    else:
        # cat
        owner[(playerNumber+edge.construction.owner)%4] = 1
    return owner



hexInNodeLowerBound = np.array([*otherInfoLowerBound, *nodesLowerBound, *edgesLowerBound])
hexInNodeUpperBound = np.array([*otherInfoUpperBound, *nodesUpperBound, *edgesUpperBound])
# 1548
def getHexInNodeObs(gameState: GameState, playerNumber=0):
    # Other info
    otherInfo = getOtherInfo(gameState, playerNumber)
    
    # Get node info containing hexes
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentation(nodes[nodeIndex], gameState, playerNumber=playerNumber))

    # Get edge info
    edgeInfo = []
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState, playerNumber=playerNumber))

    output = [*otherInfo, *nodeInfo, *edgeInfo]
    return np.array(output)


nodeInHexLowerBound = np.array([*otherInfoLowerBound, *hexsLowerBound, *edgesLowerBound])
nodeInHexUpperBound = np.array([*otherInfoUpperBound, *hexsLowerBound, *edgesUpperBound])
# 2257
def getNodeInHexObs(gameState: GameState, playerNumber=0):
    # Other info
    otherInfo = getOtherInfo(gameState, playerNumber)
    
    # Get node info containing hexes
    hexes = gameState.boardHexes
    hexInfo = []
    for hexIndex in constructableHexesList:
        hexInfo.extend(getHexRepresentation(hexes[hexIndex], gameState, playerNumber=playerNumber))

    # Get edge info
    edgeInfo = [] # 144
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState, playerNumber=playerNumber))

    output = [*otherInfo, *hexInfo, *edgeInfo]
    return np.array(output)