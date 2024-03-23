from Game.CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanPlayer import Player
from Game.CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from Game.CatanAction import *
import numpy as np

def getSetupObservation(gameState: GameState, playerNumber=0):
    """
    returns dotTotal for each node (54)
    """

    # Get Node info
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getSetupNodeRepresentation(nodes[nodeIndex], gameState))
    
    return np.array(nodeInfo)

def getSetupObservationValue(gameState: GameState, playerNumber=0):
    """
    returns single integer with value of node - production/diversity/ports,
    """
    # Get Node info
    nodes = gameState.boardNodes
    nodeInfo = []
    for nodeIndex in constructableNodesList:
        nodeInfo.append(getCompactNodeInfo(nodes[nodeIndex], gameState))
    
    return np.array(nodeInfo)

getNodeValueLowerBounds = np.array(54*[0])
getNodeValueUpperBounds = np.array(54*[16])
def getCompactNodeInfo(node: BoardNode, gameState: GameState):
    """
    Return 2 values - 1st is a general value of node taking production/port/diversity into account, 2nd is 1-hot vector 10 if wood-brick, 01 if ore-wheat, 00 if neither.
    """
    value = 0
    # Get node production
    productionList = [0, 0, 0, 0, 0]
    adjTileNumbers = node.GetAdjacentHexes()
    adjTiles = [gameState.boardHexes[tileNumber] for tileNumber in adjTileNumbers if tileNumber != None]
    for tile in adjTiles:
        if tile.production == None:
            continue
        productionList[resourceIndex[tile.production]] += numberDotsMapping[tile.number]
    # Port type
    portType = [0, 0, 0, 0, 0, 0, 0]
    portType[portTypeIndex[node.portType]] = 1
    # Diversity
    resources = [1 if num != 0 else 0 for num in productionList]

    value += sum(productionList)
    value += sum(resources)

    return value

def getNodeValue(node: BoardNode, gameState: GameState):
    # Get node production
    productionList = [0, 0, 0, 0, 0]
    adjTileNumbers = node.GetAdjacentHexes()
    adjTiles = [gameState.boardHexes[tileNumber] for tileNumber in adjTileNumbers if tileNumber != None]
    for tile in adjTiles:
        if tile.production == None:
            continue
        productionList[resourceIndex[tile.production]] += numberDotsMapping[tile.number]
    production = sum(productionList)
    # Port type
    portType = [0, 0, 0, 0, 0, 0, 0]
    portType[portTypeIndex[node.portType]] = 1
    # Diversity
    resources = [1 if num != 0 else 0 for num in productionList]

    # # 2:1 port with matching resources total >= 4
    # for i in range(5):
    #     if portType[i] == 1 and productionList[i] >= 4:
    #         value = 5 * productionList[i] # (4 - 9)
    #         return value
    # # 3:1 port with resources total >= 8
    # if portType[5] == 1 and sum(productionList) >= 8:
    #     value = 2 * sum(productionList) # (16-19)
    #     return value
    
    # # If it doesnt get a port then production and diversity are most important, production(0-13) diversity(0-3), (0 - 19)
    # value = production + (2 * diversity)
    # if value <= 10:
    #     return 0
    # else:
    #     return value
    value = 0
    value += sum(productionList)
    value += sum(resources)

    return value


setupRandomLowerBounds = np.array(54 * (22*[0]))
var = ([1] * 16) + (5 * [5]) + [15]
setupRandomUpperBounds = np.array(54 * var)
def getSetupRandomObservation(gameState: GameState, playerNumber=0):
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
def getSetupRandomWithRoadsObservation(gameState: GameState, playerNumber=0):
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
    "WAITING_FOR_TRADE":    [0, 0, 0, 1, 0, 0, 0, 0],
}

phaseOneHotMappingFull = {
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
    "OVER":                 [0, 0, 0, 0, 0, 0, 0, 0],
}

# returns 2350 length 1D array (now 2358)
def getObservation(gameState: GameState, playerPosition=0):
    """
    Returns all game info for a model (2358)
    """

    player:Player = gameState.players[playerPosition]
    if len(gameState.players) > 1:
        player1 = gameState.players[(playerPosition+1)%4]
        player2 = gameState.players[(playerPosition+2)%4]
        player3 = gameState.players[(playerPosition+3)%4]

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
    try:
        player1VP = getVisibleVictoryPoints(player1)
        player2VP = getVisibleVictoryPoints(player2)
        player3VP = getVisibleVictoryPoints(player3)
    except Exception as e:
        player1VP = 0
        player2VP = 0
        player3VP = 0


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
def getObservationNoPhase(gameState: GameState, playerNumber=0):
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
roadCountUpper = [15]
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
    # possibleRoads = player.possibleRoads (num possible roads)
    # possibleSettlements = player.possibleSettlements (num possible settlement nodes)


##################################################################################################################################################################################################################################################


hexRobberInfoLower = [-1] * 19
hexRobberInfoUpper = [60] * 19

singleNodeInfoLower = [0] * 20
# [*owner, *constructionType, *portType, *dotList, dotTotal, canBuildSettlement]
singleNodeInfoUpper = ([1] * 13) + ([15]*5) + [15] + [1]# + ([1]*5)
nodeInfoLowerSimplified = singleNodeInfoLower * 54
nodeInfoUpperSimplified = singleNodeInfoUpper * 54

edgeInfoLowerSimplified = [0, 0, 0, 0] * 72
edgeInfoUpperSimplified = [1, 1, 1, 1] * 72


lowerBoundsSimplified = np.array([
    *myResourcesLower,
    *developmentCardsLower,
    *myVictoryPointsLower,
    *moreThan7ResourcesLower,
    *tradeRatesLower,
    *knightsLower,
    *roadCountLower,
    0, #CanAffordSettlement
    0, #CanAffordCity
    0, #CanAffordRoad
    0, #*longestRoadPlayerLower,
    0, #*largestArmyPlayerLower,
    *nodeInfoLowerSimplified,
    *hexRobberInfoLower,
    *edgeInfoLowerSimplified,
    *phaseLower
])

upperBoundsSimplified = np.array([
    *myResourcesUpper,
    *developmentCardsUpper,
    *myVictoryPointsUpper,
    *moreThan7ResourcesUpper,
    *tradeRatesUpper,
    *knightsUpper,
    *roadCountUpper,
    1, #CanAffordSettlement
    1, #CanAffordCity
    1, #CanAffordRoad
    1, # *longestRoadPlayerUpper,
    1, # *largestArmyPlayerUpper,
    *nodeInfoUpperSimplified,
    *hexRobberInfoUpper,
    *edgeInfoUpperSimplified,
    *phaseUpper
])

def getObservationSimplified(gameState: GameState, playerNumber=0):
    """
    Simplified obervation with more useful features - 1492
    """
    player:Player = gameState.players[playerNumber]

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
    longestRoadPlayer = 1 if gameState.longestRoadPlayer == playerNumber else 0 # 1
    largestArmyPlayer = 1 if gameState.largestArmyPlayer == playerNumber else 0 # 1

    # TODO:
    # Add HaveBuiltFirstSettlement, availableSettlements, availableRoads, availableCity, CanAffordDevCard, CurrentProduction, NumSettlements(including cities), NumCities, NumRoads, NumDevCardsBought,

    ## Board info ##

    # Get hex robber info
    hexes = gameState.boardHexes
    hexInfo = [] # 19
    for hexIndex in constructableHexesList:
        hexInfo.append(getHexRobberRating(hexes[hexIndex], gameState, playerNumber=playerNumber))
    
    # Get node info
    nodes = gameState.boardNodes
    nodeInfo = [] # 54 * 20 = 1080
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentationSimilified(nodes[nodeIndex], gameState, playerNumber=playerNumber))

    # Get edge info
    edgeInfo = [] # 72 * 4 = 288
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentationSimplified(gameState.boardEdges[edgeIndex], gameState, playerNumber=playerNumber))

    # Get Game phase
    phase = phaseOneHotMapping[gameState.currState] # 8

    output = [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, roadCount, canAffordSettlement, canAffordCity, canAffordRoad, longestRoadPlayer, largestArmyPlayer, *nodeInfo, *hexInfo, *edgeInfo, *phase]
    return np.array(output)

def getHexRobberRating(hex: BoardHex, gameState: GameState, playerNumber=0) -> int:
    """
    Number representing whether to play robber: shouldn't place on own buildings - 1 for each hex = 19
    """
    numSettlements = 0
    numCities = 0
    nodeIndexes = hex.GetAdjacentNodes()
    for nodeIndex in nodeIndexes:
        node = gameState.boardNodes[nodeIndex]
        if node.construction == None:
            continue
        # Bad rating don't place here
        if node.construction.owner == playerNumber:
            return -1
        if node.construction.type == 'SETTLEMENT':
            numSettlements += 1
        elif node.construction.type == 'CITY':
            numCities += 1     
    rating = (numSettlements * numberDotsMapping[hex.number]) + 2*(numCities * numberDotsMapping[hex.number])
    return rating

def getNodeRepresentationSimilified(node: BoardNode, gameState: GameState, playerNumber=0) -> list:
    """
    For each node get: owner, constructionType, portType, dotList, production (20 total) - 20*54 = 1080
    """
    owner = [0, 0, 0]
    # nodes cannot have roads - settlement, city, None
    constructionType = [0, 0, 0]
    portType = [0, 0, 0, 0, 0, 0, 0]

    if node.construction != None:
        owner = [1, 0, 0] if node.construction.owner == playerNumber else [0, 1, 0]
        constructionType[constructionTypeIndex[node.construction.type]-1] = 1
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
    # resourceList = [1 if x != 0 else 0 for x in dotList]

    setupPhase = not gameState.setupDone
    canBuildSettlement = int(gameState.CanBuildSettlement(gameState.players[playerNumber], node, setUpPhase=setupPhase))

    #       cat    cat               cat       num          num
    return [*owner, *constructionType, *portType, *dotList, dotTotal, canBuildSettlement] #, *resourceList]

def getEdgeRepresentationSimplified(edge: BoardEdge, gameState: GameState, playerNumber=0) -> list:
    """
    For each edge get owner - 4 * 72 = 288
    """
    owner = [0, 0, 0]
    if edge.construction == None:
        owner[-1] = 1
    else:
        owner = [1, 0, 0] if (edge.construction.owner == playerNumber) else [0, 1, 0]

    setupPhase = not gameState.setupDone
    canBuildRoad = int(gameState.CanBuildRoad(gameState.players[playerNumber], edge, edge.index, setUpPhase=setupPhase))

    return [*owner, canBuildRoad]


####################################################################################################################################################################














singleNodeInfoLowerTrading = [0] * 14
# *portType, *dotList, dotTotal, canBuildSettlement]
singleNodeInfoUpperTrading = (7*[1]) + (5*[15]) + [15] + [1]
nodeInfoLowerTrading = singleNodeInfoLowerTrading * 54
nodeInfoUpperTrading = singleNodeInfoUpperTrading * 54

edgeInfoLowerTrading = [0, 0] * 72
edgeInfoUpperTrading = [1, 1] * 72

resourceProductionLower = [0, 0, 0, 0, 0]
resourceProductionUpper = [40, 40, 40, 40, 40]



lowerBoundsTrading = np.array([
    *myResourcesLower,
    *developmentCardsLower,
    *myVictoryPointsLower,
    *moreThan7ResourcesLower,
    *tradeRatesLower,
    *knightsLower,
    *roadCountLower,
    0, # *longestRoadPlayerUpper,
    0, # *largestArmyPlayerUpper,
    0, #CanAffordSettlement
    0, #CanAffordCity
    0, #CanAffordRoad
    0, # CanAffordDevCard
    0, # PossibleSettlement
    0, # PossibleCity
    0, # NumSettlementsBuilt
    0, # NumCitiesBuilt
    0, # Num roads built
    0, # HaveBuilt1stSettlement
    0, # HaveBuilt1stCity,
    *resourceProductionLower, # Resource Production
    *nodeInfoUpperTrading,
    *hexRobberInfoUpper,
    *edgeInfoUpperTrading,
    *phaseUpper
])

upperBoundsTrading = np.array([
    *myResourcesUpper,
    *developmentCardsUpper,
    *myVictoryPointsUpper,
    *moreThan7ResourcesUpper,
    *tradeRatesUpper,
    *knightsUpper,
    *roadCountUpper,
    1, # *longestRoadPlayerUpper,
    1, # *largestArmyPlayerUpper,
    1, #CanAffordSettlement
    1, #CanAffordCity
    1, #CanAffordRoad
    1, # CanAffordDevCard
    1, # PossibleSettlement
    1, # PossibleCity
    3, # NumSettlementsBuilt
    4, # NumCitiesBuilt
    13, # Num roads built
    1, # HaveBuilt1stSettlement
    1, # HaveBuilt1stCity,
    *resourceProductionUpper, # Resource Production
    *nodeInfoUpperTrading,
    *hexRobberInfoUpper,
    *edgeInfoUpperTrading,
    *phaseUpper
])

def getObservationTrading(gameState: GameState, playerNumber=0):
    """
    Final tuned observation for trading - 965
    """
    player:Player = gameState.players[playerNumber]

    ## My info ##
    myResources = player.resources # 6
    developmentCards = player.developmentCards # 5
    myVictoryPoints = player.victoryPoints # 1
    moreThan7Resources = int(len(myResources) > 7) # 1
    tradeRates = player.tradeRates # 5
    # Number of used knights
    knights = player.knights # 1
    roadCount = player.roadCount # 1
    # Useful Info for trading
    longestRoadPlayer = 1 if gameState.longestRoadPlayer == playerNumber else 0 # 1
    largestArmyPlayer = 1 if gameState.largestArmyPlayer == playerNumber else 0 # 1
    canAffordSettlement = int(player.CanAfford(BuildSettlementAction.cost)) # 1
    canAffordCity = int(player.CanAfford(BuildCityAction.cost)) # 1
    canAffordRoad = int(player.CanAfford(BuildRoadAction.cost)) # 1
    canAffordDevelopmentCard = int(player.CanAfford(BuyDevelopmentCardAction.cost)) # 1
        # Could build if we can afford
    possibleSettlement = 0
    if gameState.currState[:5] != "START":
        if gameState.GetPossibleSettlements(player):
            possibleSettlement = 1  
    possibleCity = 1 if player.settlements else 0
    numSettlementsBuilt = len(player.settlements) + len(player.cities) - 2 # excluding setup - 1
    numCitiesBuilt = len(player.cities) # 1
    numRoadsBuilt = len(player.roads) - 2 # excluding setup - 1
    haveBuilt1stSettlement = int(numSettlementsBuilt > 0) # 1
    haveBuilt1stCity = int(numCitiesBuilt > 0) # 1
    
    # Current Production
    resourceProduction = listm([0, 0, 0, 0, 0, 0]) # 5
    for diceNumber, resourceList in player.diceProduction.items():
        resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
    resourceProduction = resourceProduction[:-1]

    # 38 ^^
    ## Board info ##

    # Get hex robber info
    hexes = gameState.boardHexes
    hexInfo = [] # 19
    for hexIndex in constructableHexesList:
        hexInfo.append(getHexRobberRating(hexes[hexIndex], gameState, playerNumber=playerNumber))
    
    # Get node info
    nodes = gameState.boardNodes
    nodeInfo = [] # 756
    for nodeIndex in constructableNodesList:
        nodeInfo.extend(getNodeRepresentationTrading(nodes[nodeIndex], gameState, playerNumber=playerNumber))

    # Get edge info
    edgeInfo = [] # 144
    for edgeIndex in constructableEdgesList:
        edgeInfo.extend(getEdgeRepresentationTrading(gameState.boardEdges[edgeIndex], gameState, playerNumber=playerNumber))

    # Get Game phase
    phase = phaseOneHotMapping[gameState.currState] # 8

    output = [*myResources, *developmentCards, myVictoryPoints, moreThan7Resources, *tradeRates, knights, roadCount, longestRoadPlayer, largestArmyPlayer, canAffordSettlement, canAffordCity, canAffordRoad, canAffordDevelopmentCard, possibleSettlement, possibleCity, numSettlementsBuilt, numCitiesBuilt, numRoadsBuilt, haveBuilt1stSettlement, haveBuilt1stCity, *resourceProduction, *nodeInfo, *hexInfo, *edgeInfo, *phase]
    return np.array(output)

def getNodeRepresentationTrading(node: BoardNode, gameState: GameState, playerNumber=0) -> list:
    """
    For each node get: portType, dotList, production, canBuild (14 total) - 14*54 = 756
    """
    portType = [0, 0, 0, 0, 0, 0, 0]
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
    # resourceList = [1 if x != 0 else 0 for x in dotList]

    setupPhase = not gameState.setupDone
    canBuildSettlement = int(gameState.CanBuildSettlement(gameState.players[playerNumber], node, setUpPhase=setupPhase))

    #       cat       num          num      cat
    return [*portType, *dotList, dotTotal, canBuildSettlement]

def getEdgeRepresentationTrading(edge: BoardEdge, gameState: GameState, playerNumber=0) -> list:
    """
    For each edge return if we own it and can we build road on it - 2 * 72 = 144
    """
    myEdge = 0
    if edge.construction != None:
        if edge.construction.owner == playerNumber:
            myEdge = 1

    setupPhase = not gameState.setupDone
    canBuildRoad = int(gameState.CanBuildRoad(gameState.players[playerNumber], edge, edge.index, setUpPhase=setupPhase))

    return [myEdge, canBuildRoad]