from Game.CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanPlayer import Player
from Game.CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from Game.CatanAction import *
from DeepLearning.GetObservation import getVisibleVictoryPoints, phaseOneHotMapping
import numpy as np

otherInfoLowerBound = 55*[0]
otherInfoUpperBound = 5*[10] + 5*[10] + [10] + [1] + 5*[4] + [10] + 5*[1] + [15] + 5*[1] + 9*[1] + 5*[40] + 3*[1] + 8*[1] + [150]
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

    playerTurns = player.playerTurns # 1

    return [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, *largestArmyPlayer, roadCount, *longestRoadPlayer, canAffordSettlement, canAffordCity, canAffordRoad, canAffordDevelopmentCard, possibleSettlement, possibleCity, numSettlementsBuilt, numCitiesBuilt, numRoadsBuilt, *resourceProduction, player1VP, player2VP, player3VP, *phase, playerTurns]


hexLowerBound = [0]*8
hexUpperBound = [1] + [5] + 6*[1]
hexsLowerBound = hexLowerBound * 19
hexsUpperBound = hexUpperBound * 19
def getHexRepresentation(hex: BoardHex, gameState: GameState, playerNumber=0) -> list:
    """
    For each hex get number, resource, for each surrounding node get: owner, construction type, 19 * 8 = 152
    """
    dot = numberDotsMapping[hex.number]
    resource = [0, 0, 0, 0, 0, 0]
    resource[resourceIndex[hex.production]] = 1

    robber = int(gameState.robberPos == hex.index)
    #       1        1     6
    return [robber, dot, *resource]


nodeLowerBound = 15*[0]
nodeUpperBound = 15*[1]
nodesLowerBound = nodeLowerBound*54
nodesUpperBound = nodeUpperBound*54
def getNodeRepresentation(node: BoardNode, gameState: GameState, playerNumber=0) -> list:
    """
    For each node get: 54*15 = 810
    """
    owner = [0, 0, 0, 0, 0]
    constructionType = [0, 0, 0]
    portType = [0, 0, 0, 0, 0, 0, 0]

    if node.construction != None:
        owner[(playerNumber+node.construction.owner)%4] = 1
        constructionType[constructionTypeIndex[node.construction.type]-1] = 1
    else:
        owner[-1] = 1
        constructionType[-1] = 1

    portType[portTypeIndex[node.portType]] = 1

    #       5            3                7            
    return [*owner, *constructionType, *portType]



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


lowerBound = np.array([*otherInfoLowerBound, *hexsLowerBound, *nodesLowerBound, *edgesLowerBound])
upperBound = np.array([*otherInfoUpperBound, *hexsLowerBound, *nodesUpperBound, *edgesUpperBound])
# 1804
def getObservationFull(gameState: GameState, playerNumber=0):
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