from Game.CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanPlayer import Player
from Game.CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from Game.CatanAction import *
from DeepLearning.GetObservation import getVisibleVictoryPoints, phaseOneHotMapping
import numpy as np

otherInfoLowerBound = 39*[0]
otherInfoUpperBound = 5*[10] + 5*[10] + [10] + [1] + 5*[4] + [10] + [1] + [15] + [1] + 9*[1] + 5*[40] + 3*[1] + [7]
# Total: 46
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
    largestArmyPlayer = gameState.largestArmyPlayer # 1
    roadCount = player.roadCount # 1
    longestRoadPlayer = gameState.longestRoadPlayer # 1
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
    phase = phaseOneHotMapping[gameState.currState]
    phase = phase.index(1) # 1

    return [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, largestArmyPlayer, roadCount, longestRoadPlayer, canAffordSettlement, canAffordCity, canAffordRoad, canAffordDevelopmentCard, possibleSettlement, possibleCity, numSettlementsBuilt, numCitiesBuilt, numRoadsBuilt, *resourceProduction, player1VP, player2VP, player3VP, phase]

hexLowerBound = [0]*20
hexUpperBound = [5] + [5] + 6*[4, 2, 6]
hexsLowerBound = hexLowerBound * 19
hexsUpperBound = hexUpperBound * 19
def getHexRepresentation(hex: BoardHex, gameState: GameState, playerNumber=0) -> list:
    """
    For each hex get number, resource, for each surrounding node get: owner, construction type, 19 * 20 = 380
    """
    dot = numberDotsMapping[hex.number]
    resource = resourceIndex[hex.production]
    
    # For each adjacent node add owner, construction type, port type
    adjNodesInfo = []
    nodeIndexes = hex.GetAdjacentNodes()
    for nodeIndex in nodeIndexes:
        owner = 0
        constructionType = 0
        portType = 0
        node = gameState.boardNodes[nodeIndex]
        portType = portTypeIndex[node.portType]
        if node.construction == None:
            owner = 4
            constructionType = 2
        else:
            owner = (playerNumber+node.construction.owner)%4
            constructionType = constructionTypeIndex[node.construction.type]-1
        adjNodesInfo.extend([owner, constructionType, portType])
    #       1,   1,          3*6
    return [dot, resource, *adjNodesInfo]



edgeLowerBound = [0]
edgeUpperBound = [4]
edgesLowerBound = 72*edgeLowerBound
edgesUpperBound = 72*edgeUpperBound
def getEdgeRepresentation(edge: BoardEdge, gameState: GameState, playerNumber=0) -> list:
    """
    For each edge get owner, 1 - 72
    """
    owner = 0
    if edge.construction == None:
        owner = 4
    else:
        # cat
        owner = (playerNumber+edge.construction.owner)%4
    return owner


labelLowerBound = np.array([*otherInfoLowerBound, *hexsLowerBound, *edgesLowerBound])
labelUpperBound = np.array([*otherInfoUpperBound, *hexsLowerBound, *edgesUpperBound])
# 2257
def getLabelObs(gameState: GameState, playerNumber=0):
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
        edgeInfo.append(getEdgeRepresentation(gameState.boardEdges[edgeIndex], gameState, playerNumber=playerNumber))

    output = [*otherInfo, *hexInfo, *edgeInfo]
    return np.array(output)