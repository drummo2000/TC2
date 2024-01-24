from CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from CatanPlayer import Player
from CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from CatanAction import *
from itertools import combinations
import math
import random

    


class AgentPolicy(Player):

    def __init__(self, name, seatNumber):

        super(AgentPolicy, self).__init__(name, seatNumber)
        self.agentName              = name
    
    def GetAllPossibleActions_RegularTurns(self, gameState: GameState, player: Player):

        possibleActions     = []

        if player.settlements and\
            player.HavePiece(g_pieces.index('CITIES')) and\
            player.CanAfford(BuildCityAction.cost):

            possibleActions += [BuildCityAction(player.seatNumber, pos, len(player.cities)) for pos in
                                gameState.GetPossibleCities(player)]

        if player.HavePiece(g_pieces.index('SETTLEMENTS')) and \
            player.CanAfford(BuildSettlementAction.cost):

            possibleActions += [BuildSettlementAction(player.seatNumber, pos, len(player.settlements)) for pos in
                                gameState.GetPossibleSettlements(player)]

        if player.HavePiece(g_pieces.index('ROADS')) and \
            player.CanAfford(BuildRoadAction.cost):

            possibleActions += [BuildRoadAction(player.seatNumber, pos, len(player.roads)) for pos in
                                gameState.GetPossibleRoads(player)]

        if gameState.CanBuyADevCard(player) and not player.biggestArmy:
            possibleActions.append(BuyDevelopmentCardAction(player.seatNumber))

        if not player.playedDevCard and sum(player.developmentCards[:-1]) > 0:

            possibleCardsToUse = []
            if not player.playedDevCard:

                if player.developmentCards[MONOPOLY_CARD_INDEX] > 0 and \
                        player.mayPlayDevCards[MONOPOLY_CARD_INDEX]:
                    possibleCardsToUse += self.GetMonopolyResource(gameState, player)

                if player.developmentCards[YEAR_OF_PLENTY_CARD_INDEX] > 0 and \
                        player.mayPlayDevCards[YEAR_OF_PLENTY_CARD_INDEX]:
                    possibleCardsToUse += self.GetYearOfPlentyResource(gameState, player)

                if player.developmentCards[ROAD_BUILDING_CARD_INDEX] > 0 and \
                        player.mayPlayDevCards[ROAD_BUILDING_CARD_INDEX] and \
                        player.numberOfPieces[0] > 0:
                    possibleCardsToUse += [UseFreeRoadsCardAction(player.seatNumber, None, None)]

            if possibleCardsToUse:
                possibleActions += possibleCardsToUse

        possibleTrade = self.GetPossibleBankTrades(gameState, player)
        if possibleTrade is not None and possibleTrade:
            possibleActions.append(possibleTrade[int(random.random() * len(possibleTrade))])

        possibleActions.append(EndTurnAction(playerNumber=player.seatNumber))

        return possibleActions
    
    def GetAllPossibleActions_SpecialTurns(self, gameState: GameState, player: Player):

        if gameState.currState == 'PLACING_ROBBER':

            # Rolled out 7  * or *  Used a knight card
            return self.ChooseRobberPosition(gameState, player)

        elif gameState.currState == 'WAITING_FOR_DISCARDS':

            return self.ChooseCardsToDiscard()

        elif gameState.currState == 'WAITING_FOR_CHOICE':

            return self.ChoosePlayerToStealFrom(gameState, player)

        elif gameState.currState == "PLACING_FREE_ROAD1":

            possibleRoads = gameState.GetPossibleRoads(player)

            if possibleRoads is None or not possibleRoads or player.numberOfPieces[0] <= 0:
                return [ ChangeGameStateAction("PLAY1") ]

            return [BuildRoadAction(player.seatNumber, roadEdge,
                                    len(player.roads))
                    for roadEdge in possibleRoads]

        elif gameState.currState == "PLACING_FREE_ROAD2":

            possibleRoads = gameState.GetPossibleRoads(player)

            if possibleRoads is None or not possibleRoads or player.numberOfPieces[0] <= 0:
                return [ ChangeGameStateAction("PLAY1") ]

            return [BuildRoadAction(player.seatNumber, roadEdge,
                                    len(player.roads))
                    for roadEdge in possibleRoads]

        elif gameState.currState == "WAITING_FOR_TRADE":

            return RejectTradeOfferAction(playerNumber=player.seatNumber)

    # Returns list of possible actions in given state
    def GetPossibleActions(self, gameState, player = None):

        if player is None:
            player = self

        # Call function based on game phase
        if not gameState.setupDone:
            return self.GetAllPossibleActions_Setup(gameState, player)
        elif gameState.currState == "PLAY":
            return self.GetAllPossibleActions_PreDiceRoll(player)
        elif gameState.currState == "PLAY1":
            return self.GetAllPossibleActions_RegularTurns(gameState, player)
        else:
            return self.GetAllPossibleActions_SpecialTurns(gameState, player)


    # Return selected action
    def DoMove(self, game):

        # If not my turn and were not in WAITING_FOR_DISCARDS phase then return None
        if game.gameState.currPlayer != self.seatNumber and \
            game.gameState.currState != "WAITING_FOR_DISCARDS":
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        if possibleActions is not None:
            if type(possibleActions) == list:
                return random.choice(possibleActions)
            else:
                return possibleActions

        raise("My agent has no possible actions returning NONE")
        # NOTE: If no actions returned should we return EndTurn/RollDice?
        return None

    # TODO: later if discarding more than 4 resources, choose first 4 then the rest choose randomly
    def ChooseCardsToDiscard(self):
        player = self
        
        if sum(player.resources) <= 7:
            return DiscardResourcesAction(player.seatNumber, [0, 0, 0, 0, 0, 0])
            
        resourcesPopulation =   [0 for i in range(0, player.resources[0])] + \
                                [1 for j in range(0, player.resources[1])] + \
                                [2 for k in range(0, player.resources[2])] + \
                                [3 for l in range(0, player.resources[3])] + \
                                [4 for m in range(0, player.resources[4])] + \
                                [5 for n in range(0, player.resources[5])]

        possibleSelections = []
        if sum(player.resources) <= 11:
            # Returns list of tuples
            possibleSelections = set(combinations(resourcesPopulation, len(resourcesPopulation)//2))
            return [DiscardResourcesAction(player.seatNumber, [selection.count(0),
                                                                selection.count(1),
                                                                selection.count(2),
                                                                selection.count(3),
                                                                selection.count(4),
                                                                selection.count(5)])
                        for selection in possibleSelections]
        # if more than 11 resources just return random selection
        else:
            discardCardCount = int(math.floor(len(resourcesPopulation) / 2.0))

            selectedResources = random.sample(resourcesPopulation, discardCardCount)

            return DiscardResourcesAction(player.seatNumber, [selectedResources.count(0),
                                                            selectedResources.count(1),
                                                            selectedResources.count(2),
                                                            selectedResources.count(3),
                                                            selectedResources.count(4),
                                                            selectedResources.count(5)])

    # Function which returns PlaceRobberActions for all hex positions
    def ChooseRobberPosition(self, gameState: GameState, player: Player) -> list:
        possiblePositions = gameState.possibleRobberPos
        return [PlaceRobberAction(player.seatNumber, pos) for pos in possiblePositions]

    def ChoosePlayerToStealFrom(self, gameState, player):

        if player is None:
            player = self
        possiblePlayers = gameState.GetPossiblePlayersToSteal(player.seatNumber)
        if len(possiblePlayers) > 0:
            return [ChoosePlayerToStealFromAction(player.seatNumber, possiblePlayer) for possiblePlayer in possiblePlayers]
        return None

    def GetPossibleBankTrades(self, gameState, player):

        if player is None:
            player = self

        possibleTrades = []

        for i in range(5):
            if int(player.resources[i] / self.tradeRates[i]) > 0:
                give = [0, 0, 0, 0, 0]
                give[i] = self.tradeRates[i]
                for j in range(1, 5):
                    get = [0, 0, 0, 0, 0]
                    index = (i + j) % 5
                    get[index] = 1
                    possibleTrades.append(BankTradeOfferAction(player.seatNumber, give, get))

        return possibleTrades


    def GetMonopolyResource(self, game, player):

        if player is None:
            player = self

        return [ UseMonopolyCardAction(player.seatNumber, resource) for resource in range(5) ]


    def GetYearOfPlentyResource(self, game, player):

        if player is None:
            player = self

        actionPossibilities = []

        for i in range(0, 5):
            for j in range(i, 5):
                chosenResources = [0, 0, 0, 0, 0]
                chosenResources[i] += 1
                chosenResources[j] += 1
                actionPossibilities.append(UseYearOfPlentyCardAction(player.seatNumber, chosenResources))

        return actionPossibilities





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
    return output

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