from Game.CatanAction import *
import math
import sys
from Game.CatanUtilsPy import CanAfford as cf
from Game.CatanUtilsPy import listm
from Game.CatanBoard import g_resources

class PlayerStats(object):

    def __init__(self):

        # General
        self.numTurns = 0 # Every time endTurn is called not each action
        self.victoryPoints = 0
        self.numRoadsBuilt = 0
        # Dev card Breakdown
        self.devCardsBought = 0
        self.usedDevCards = listm([0, 0, 0, 0, 0])
        # Point Breakdown
        self.settlementsBuilt      = 0
        self.citiesBuilt           = 0
        self.devCardVP             = 0
        self.largestArmy           = 0
        self.longestRoad           = 0
        # Resource Breakdown
        self.resourcesReceived = listm([0, 0, 0, 0, 0])
        self.totalResourcesDiscarded = 0 # from rolling 7
        self.totalResourcesStolen  = 0 # from monopoly/knight (stolen from)
        self.resourcesFromDevCard = listm([0, 0, 0, 0, 0]) # from monopoly/year of plenty
        self.resourcesFromBankTrade = listm([0, 0, 0, 0, 0])
        self.finalResourceProduction = listm([0, 0, 0, 0, 0, 0])
        self.finalTradeRates = listm([0, 0, 0, 0, 0])
        # Setup Breakdown
        self.setupResourceProduction = listm([0, 0, 0, 0, 0, 0])
        self.setupTradeRates = listm([0, 0, 0, 0, 0])
        self.setupResourceDiversity = 0
    
    def __str__(self):
        output = f"General\n" + \
                 f"    numTurns: {self.numTurns}\n" + \
                 f"    victoryPoints: {self.victoryPoints}\n" + \
                 f"    finalTradeRates: {self.finalTradeRates}\n" + \
                 f"    numRoadsBuilt: {self.numRoadsBuilt}\n" + \
                 f"Dev Card Breakdown\n" + \
                 f"    devCardsBought: {self.devCardsBought}\n" + \
                 f"    usedDevCards: {self.usedDevCards}\n" + \
                 f"Point Breakdown\n" + \
                 f"    settlementsBuilt: {self.settlementsBuilt}\n" + \
                 f"    citiesBuilt: {self.citiesBuilt}\n" + \
                 f"    devCardVP: {self.devCardVP}\n" + \
                 f"    largestArmy: {self.largestArmy}\n" + \
                 f"    longestRoad: {self.longestRoad}\n" + \
                 f"Resource Breakdown\n" + \
                 f"    resourcesReceived: {self.resourcesReceived}\n" + \
                 f"    totalResourcesReceivedPerTurn: {round(sum(self.resourcesReceived)/self.numTurns, 3)}\n" + \
                 f"    totalResourcesDiscarded: {self.totalResourcesDiscarded}\n" + \
                 f"    totalResourcesStolen: {self.totalResourcesStolen}\n" + \
                 f"    resourcesFromDevCard: {self.resourcesFromDevCard}\n" + \
                 f"    totalResourcesFromDevCard: {round(sum(self.resourcesFromDevCard), 3)}\n" + \
                 f"    resourcesFromBankTrade: {self.resourcesFromBankTrade}\n" + \
                 f"    totalResourcesFromBankTrade: {round(sum(self.resourcesFromBankTrade), 3)}\n" + \
                 f"    finalResourceProduction: {self.finalResourceProduction[:-1]}\n" + \
                 f"Setup Breakdown\n" + \
                 f"    setupResourceProduction: {self.setupResourceProduction[:-1]}\n" + \
                 f"    totalSetupResourceProduction: {round(sum(self.setupResourceProduction[:-1]))}\n" + \
                 f"    setupTradeRates: {self.setupTradeRates}\n" + \
                 f"    setupResourceDiversity: {self.setupResourceDiversity}\n" + \
                 f"RESOURCES: {g_resources}\n"
        return output
    
    def getList(self):
        # 22
        return [
                self.numTurns,
                self.victoryPoints,
                self.finalTradeRates,
                self.numRoadsBuilt,
                self.devCardsBought,
                self.usedDevCards,
                self.settlementsBuilt,
                self.citiesBuilt,
                self.devCardVP,
                self.largestArmy,
                self.longestRoad,
                self.resourcesReceived,
                round(sum(self.resourcesReceived)/self.numTurns, 3),
                self.totalResourcesDiscarded,
                self.totalResourcesStolen,
                self.resourcesFromDevCard,
                round(sum(self.resourcesFromDevCard), 3),
                self.resourcesFromBankTrade,
                round(sum(self.resourcesFromBankTrade), 3),
                self.finalResourceProduction[:-1],
                self.setupResourceProduction[:-1],
                round(sum(self.setupResourceProduction[:-1])),
                self.setupTradeRates,
                self.setupResourceDiversity
        ]

class PlayerStatsTracker(PlayerStats):
    def __init__(self):
        super(PlayerStatsTracker, self).__init__()
        self.numGames = 0
    
    def __add__(self, other):
        if isinstance(other, PlayerStats):
            self.numGames += 1
            # General
            self.numTurns += other.numTurns
            self.victoryPoints += other.victoryPoints
            self.numRoadsBuilt += other.numRoadsBuilt
            # Dev card Breakdown
            self.devCardsBought += other.devCardsBought
            self.usedDevCards += other.usedDevCards
            # Point Breakdown
            self.settlementsBuilt      += other.settlementsBuilt
            self.citiesBuilt           += other.citiesBuilt
            self.devCardVP             += other.devCardVP
            self.largestArmy           += other.largestArmy
            self.longestRoad           += other.longestRoad
            # Resource Breakdown
            self.resourcesReceived += other.resourcesReceived
            self.totalResourcesDiscarded += other.totalResourcesDiscarded
            self.totalResourcesStolen += other.totalResourcesStolen
            self.resourcesFromDevCard += other.resourcesFromDevCard
            self.resourcesFromBankTrade += other.resourcesFromBankTrade
            self.finalResourceProduction += other.finalResourceProduction
            self.finalTradeRates += other.finalTradeRates
            # Setup Breakdown
            self.setupResourceProduction += other.setupResourceProduction
            self.setupTradeRates += other.setupTradeRates
            self.setupResourceDiversity += other.setupResourceDiversity
            return self
        else:
            ValueError("Can only add PlayerStats objects")

    def getAverages(self):
        # General
        self.numTurns = self.numTurns / self.numGames
        self.victoryPoints = self.victoryPoints / self.numGames
        self.numRoadsBuilt = self.numRoadsBuilt / self.numGames
        # Dev card Breakdown
        self.devCardsBought = self.devCardsBought / self.numGames
        self.usedDevCards = self.usedDevCards / self.numGames 
        # Point Breakdown
        self.settlementsBuilt      = self.settlementsBuilt / self.numGames
        self.citiesBuilt           = self.citiesBuilt / self.numGames
        self.devCardVP             = self.devCardVP / self.numGames
        self.largestArmy           = self.largestArmy / self.numGames
        self.longestRoad           = self.longestRoad / self.numGames
        # Resource Breakdown
        self.resourcesReceived = self.resourcesReceived / self.numGames
        self.totalResourcesDiscarded = self.totalResourcesDiscarded / self.numGames
        self.totalResourcesStolen  = self.totalResourcesStolen / self.numGames
        self.resourcesFromDevCard = self.resourcesFromDevCard / self.numGames
        self.resourcesFromBankTrade = self.resourcesFromBankTrade / self.numGames
        self.finalResourceProduction = self.finalResourceProduction / self.numGames
        self.finalTradeRates = self.finalTradeRates / self.numGames
        # Setup Breakdown
        self.setupResourceProduction = self.setupResourceProduction / self.numGames
        self.setupTradeRates = self.setupTradeRates / self.numGames
        self.setupResourceDiversity = self.setupResourceDiversity / self.numGames
        


# Base player class to extend when building a new agent
class Player(object):

    def generatePlayerStats(self):
        """
        Set PlayerStats attributes after game
        """
        self.stats.victoryPoints = self.victoryPoints
        self.stats.settlementsBuilt = len(self.settlements)
        self.stats.citiesBuilt = len(self.cities)
        self.stats.devCardVP = self.developmentCards[VICTORY_POINT_CARD_INDEX]
        self.stats.largestArmy = int(self.biggestArmy)
        self.stats.longestRoad = int(self.biggestRoad)
        self.stats.finalTradeRates = self.tradeRates
        self.stats.numRoadsBuilt = len(self.roads)

        for diceNumber, resourceList in self.diceProduction.items():
            self.stats.finalResourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]

        self.stats.setupResourceDiversity = len([x for x in self.stats.setupResourceProduction if x != 0])


    Model = None

    def __init__(self, name, seatNumber):

        self.name             = name
        self.seatNumber       = seatNumber
        self.resources        = listm([0, 0, 0, 0, 0, 0])
        self.developmentCards = [ 0 for i in range(0, len(g_developmentCards))    ]
        self.mayPlayDevCards  = [ False for i in range(0, len(g_developmentCards))]
        self.recentCard       = [ 0 for i in range(0, len(g_developmentCards))]
        self.roads            = [ ]
        # list of positions of built settlements
        self.settlements      = [ ]
        self.cities           = [ ]
        self.biggestRoad      = False
        self.biggestArmy      = False
        self.numberOfPieces   = [ 15, 5, 4 ]
        self.knights          = 0
        self.playedDevCard    = False
        self.discardCardCount = 0
        self.tradeCount       = 0

        # keeps track of resources that are generated after a certain dice roll to save up processing time
        self.diceProduction = \
        {
            2  : listm([0, 0, 0, 0, 0, 0]),
            3  : listm([0, 0, 0, 0, 0, 0]),
            4  : listm([0, 0, 0, 0, 0, 0]),
            5  : listm([0, 0, 0, 0, 0, 0]),
            6  : listm([0, 0, 0, 0, 0, 0]),
            8  : listm([0, 0, 0, 0, 0, 0]),
            9  : listm([0, 0, 0, 0, 0, 0]),
            10 : listm([0, 0, 0, 0, 0, 0]),
            11 : listm([0, 0, 0, 0, 0, 0]),
            12 : listm([0, 0, 0, 0, 0, 0])
        }

        # keeps track of trade rates
        self.tradeRates = [4, 4, 4, 4, 4]

        self.possibleRoads         = []
        self.possibleSettlements   = []
        
        # flags for the setup phase (pre-game)
        self.firstSettlementBuild  = False
        self.secondSettlementBuild = False
        self.firstRoadBuild        = False
        self.secondRoadBuild       = False

        self.rolledTheDices        = False
        self.placedRobber          = False
        self.victoryPoints         = 0
        self.updateVictoryPoints   = False

        self.roadCount = 0
        self.agentName = "RANDOM"

        self.stats = PlayerStats()

    # @staticmethod
    # def LoadModel():

    #     with open('Models/Test06.mod', 'rb') as handle:
    #         Player.Model = pickle.load(handle)

    def UpdateTradeRates(self, gameState):

        availablePorts = self.GetPorts(gameState)

        if availablePorts[-1]:
            minTradeRate = 3
        else:
            minTradeRate = 4

        tradeRates = [minTradeRate, minTradeRate, minTradeRate, minTradeRate, minTradeRate]

        for i in range(0, len(tradeRates)):
            if availablePorts[i]:
                tradeRates[i] = 2

        self.tradeRates = tradeRates

    def GetVictoryPoints(self, forceUpdate=False):

        if self.updateVictoryPoints or forceUpdate:

            devCardPoints = self.developmentCards[VICTORY_POINT_CARD_INDEX]

            constructionPoints = len(self.settlements) + len(self.cities) * 2

            achievementPoints = 0
            if self.biggestRoad:
                achievementPoints += 2
            if self.biggestArmy:
                achievementPoints += 2

            self.victoryPoints = devCardPoints + constructionPoints + achievementPoints

            self.updateVictoryPoints = False

        return self.victoryPoints

    def GetStartingResources(self, gameState):

        for index in range(0, len(self.settlements)):

            settlement = gameState.boardNodes[self.settlements[index]]

            #its the second settlement
            if settlement.construction.index == 1:

                adjacentHexes = settlement.adjacentHexes

                for h in range(0, len(adjacentHexes)):

                    if adjacentHexes[h] is not None:

                        if gameState.boardHexes[adjacentHexes[h]].production is not None:

                            # logging.info("{0} : STARTING RESOURCE >> GAIN 1 {1}".format(
                            #     self.name, gameState.boardHexes[adjacentHexes[h]].production))

                            self.resources[g_resources.index(gameState.boardHexes[adjacentHexes[h]].production)] += 1
        

    def UpdatePlayerResources(self, gameState, diceNumber = None):

        if diceNumber is not None:

            diceProduction = self.diceProduction[diceNumber]

            if sum(diceProduction) > 0:
                num_start = len(self.resources)
                self.resources += diceProduction
                self.stats.resourcesReceived += diceProduction
                num_end = len(self.resources)
                if num_start != num_end:
                    print("\n\n\n UpdatePlayerResources changes length \n\n\n", self.seatNumber, self.resources, diceProduction)
                    raise("TEST ERROR")

    def UpdateLargestArmy(self, option):

        self.biggestArmy = option
        self.updateVictoryPoints = True

    def UpdateMayPlayDevCards(self, recentlyCardIndex = None, canUseAll = False):

        if canUseAll:
            for i in range(0, len(self.developmentCards)):
                self.mayPlayDevCards[i] = self.developmentCards[i] > 0
                self.recentCard[i]      = 0

        else:
            if recentlyCardIndex is None:
                for i in range(0, len(self.developmentCards)):
                    self.mayPlayDevCards[i] = self.developmentCards[i] > 0

            else:
                # IF ITS A VICTORY POINT - WE NEED TO UPDATE THE PLAYERS VICTORY POINT COUNT
                if recentlyCardIndex == VICTORY_POINT_CARD_INDEX:
                    self.updateVictoryPoints = True

                self.recentCard[recentlyCardIndex] += 1

                for i in range(0, len(self.developmentCards)):
                    self.mayPlayDevCards[i] = self.recentCard[i] < self.developmentCards[i]

    def CanAfford(self, price):
        return cf(self.resources, price)

    def DiscountAtRandom(self, discountCount):

        discountIndexes = []
        while discountCount > 0:
            for i in range(0, len(self.resources)):
                if self.resources[i] > 0:
                    discountIndexes.append(i)
            if len(discountIndexes) <= 0:
                discountCount = 0
            else:
                index = discountIndexes[int(random.random() * len(discountIndexes))]
                self.resources[index] -= 1
                discountCount -= 1

    def GetRemainingTrades(self, price):

        trades        = []
        giveIndexes   = []
        getIndexes    = []
        giveResources = []
        getResources  = []
        diff = self.resources - price
        if sum(diff) < 0:
            print("ERROR!")

        for i in range(0, len(diff)):
            if diff[i] > 0:
                giveIndexes.append(i)
            elif diff[i] < 0:
                getIndexes.append(i)
        while len(getIndexes) > 0:
            give = [0, 0, 0, 0, 0]
            get  = [0, 0, 0, 0, 0]
            giveIndex = int(random.random() * len(giveIndexes))
            index1 = giveIndexes[giveIndex]
            give[index1]  = 1
            diff[index1] -= 1
            if diff[index1] <= 0:
                giveIndexes.remove(index1)
            getIndex = int(random.random() * len(getIndexes))
            index2 = getIndexes[getIndex]
            get[index2]   = 1
            diff[index2] += 1
            if diff[index2] >= 0:
                getIndexes.remove(index2)

            giveResources.append(give)
            getResources.append(get)

        toPlayers = [True, True, True, True]
        toPlayers[self.seatNumber] = False

        for j in range(0, len(getResources)):
            trades.append(MakeTradeOfferAction(self.seatNumber, toPlayers, giveResources[j], getResources[j]))

        return trades

    def HavePiece(self, pieceIndex):
        if self.numberOfPieces[pieceIndex] > 0:
            return True

        return False

    def GetPorts(self, gameState):

        availablePorts = [ False for i in g_portType ]

        for settlementIndex in self.settlements:

            portType = gameState.boardNodes[settlementIndex].portType
            if portType is not None:
                availablePorts[g_portType.index(portType)] = True

        for cityIndex in self.cities:

            portType = gameState.boardNodes[cityIndex].portType
            if portType is not None:
                availablePorts[g_portType.index(portType)] = True

        return availablePorts

    def UpdateResourcesFromServer(self, action, element, value):

        if element in g_resources:  # RESOURCE

            if action == 'SET':
                self.resources[g_resources.index(element)] = value

            elif action == 'GAIN':
                self.resources[g_resources.index(element)] += value

            elif action == 'LOSE':

                if element == 'UNKNOWN':

                    resourceAmount = sum(self.resources) - value

                    self.resources[g_resources.index('UNKNOWN')] = resourceAmount

                    for index in range(len(self.resources) - 1):
                        self.resources[index] = 0
                else:
                    self.resources[g_resources.index(element)] -= value

            if self.resources[g_resources.index(element)] < 0:
                self.resources[g_resources.index('UNKNOWN')] += self.resources[g_resources.index(element)]
                self.resources[g_resources.index(element)] = 0

        elif element in g_pieces:  # PIECES

            if action == 'SET':
                self.numberOfPieces[g_pieces.index(element)] = value

            elif action == 'GAIN':
                self.numberOfPieces[g_pieces.index(element)] += value

            elif action == 'LOSE':
                self.numberOfPieces[g_pieces.index(element)] -= value

        elif element == 'KNIGHTS':  # KNIGHTS

            if action == 'SET':
                self.knights = value

            elif action == 'GAIN':
                self.knights += value

            elif action == 'LOSE':
                self.knights -= value

    def Build(self, gameState, pieceType, position):

        self.updateVictoryPoints = True

        if pieceType == 'ROAD':

            if not gameState.setupDone:
                gameState.UpdatePossibleRoads(self.seatNumber, pieceType, position)
            else:
                gameState.UpdatePossibleRoads(self.seatNumber, pieceType, position)
                gameState.UpdatePossibleSettlements(self.seatNumber, pieceType, position)

            if gameState.currState == "START1B":
                self.firstRoadBuild  = True
            elif gameState.currState == "START2B":
                self.secondRoadBuild = True

            newConstruction = Construction(g_constructionTypes[0],
                                           self.seatNumber, len(self.roads), position)

            if gameState.boardEdges[position].construction is not None:
                print("BOARD EDGE ALREADY CONSTRUCTED!!!!!")
                sys.exit("BOARD EDGE ALREADY CONSTRUCTED!!!!!")

            gameState.boardEdges[position].construction = newConstruction

            #gameState.constructableEdges.remove(gameState.boardEdges[position])

            self.roads.append(position)

            self.numberOfPieces[0] -= 1

        elif pieceType == 'SETTLEMENT':

            gameState.UpdatePossibleRoads(self.seatNumber, pieceType, position)
            if not gameState.setupDone:
                gameState.UpdatePossibleSettlements(self.seatNumber, pieceType, position, True)
            else:
                gameState.UpdatePossibleSettlements(self.seatNumber, pieceType, position)

            if gameState.currState == "START1A":
                self.firstSettlementBuild  = True
            elif gameState.currState == "START2A":
                self.secondSettlementBuild = True

            newConstruction = Construction(g_constructionTypes[1],
                                           self.seatNumber, len(self.settlements), position)

            if gameState.boardNodes[position].construction is not None:
                print("BOARD NODE ALREADY CONSTRUCTED!!!!!")
                print(gameState.currState)
                print(gameState.boardNodes[position].construction.owner)
                sys.exit("BOARD NODE ALREADY CONSTRUCTED!!!!!")

            gameState.boardNodes[position].construction = newConstruction

            #gameState.constructableNodes.remove(gameState.boardNodes[position])

            self.AddToDiceProduction(gameState, position)

            self.settlements.append(position)

            self.numberOfPieces[1] -= 1

            if gameState.boardNodes[position].portType is not None:
                self.UpdateTradeRates(gameState)

        elif pieceType == 'CITY':

            newConstruction = Construction(g_constructionTypes[2],
                                           self.seatNumber, len(self.cities), position)

            gameState.boardNodes[position].construction = newConstruction

            self.AddToDiceProduction(gameState, position)

            self.settlements.remove(position)

            self.cities.append(position)

            self.numberOfPieces[1] += 1

            self.numberOfPieces[2] -= 1

    def PlaceRobber(self, gameState, position):

        gameState.robberPos = position

        self.placedRobber = True

    def StartTurn(self):

        self.placedRobber   = False

        self.rolledTheDices = False

    def AddToDiceProduction(self, gameState, position):

        adjacentHexes = gameState.boardNodes[position].adjacentHexes

        for adjacentPos in adjacentHexes:

            if adjacentPos is None or gameState.robberPos == adjacentPos:
                continue

            production = gameState.boardHexes[adjacentPos].production

            if production is not None:

                number = gameState.boardHexes[adjacentPos].number

                if number is not None and number > 1:
                    self.diceProduction[number][g_resources.index(production)] += 1

    def CountRoads(self, gameState):

        def IsEdgeEmpy(edge):

            if edge is None or \
               gameState.boardEdges[edge].construction is None or \
               gameState.boardEdges[edge].construction.owner != self.seatNumber:
                return True

            return False

        startingRoads = []

        for road in self.roads:

            isStartPos = False

            front = gameState.boardEdges[road].adjacentEdges[:2]
            back  = gameState.boardEdges[road].adjacentEdges[2:]

            if IsEdgeEmpy(front[0]) and IsEdgeEmpy(front[1]):
                if road not in startingRoads:
                    startingRoads.append(road)
                isStartPos = True

            if isStartPos:
                continue

            if IsEdgeEmpy(back[0]) and IsEdgeEmpy(back[1]):
                if road not in startingRoads:
                    startingRoads.append(road)
                isStartPos = True

            if isStartPos:
                continue

            for adjacentNode in gameState.boardEdges[road].adjacentNodes:

                if adjacentNode is None:
                    continue

                if gameState.boardNodes[adjacentNode].construction is not None and \
                    gameState.boardNodes[adjacentNode].construction.owner != self.seatNumber:
                    if road not in startingRoads:
                        startingRoads.append(road)
                    break

        def DepthSearch(playerNumber, currRoad, visited, cantVisit):

            if currRoad in visited:
                return visited

            visited.append(currRoad)

            possiblePaths = []
            for adjacentEdge in gameState.boardEdges[currRoad].adjacentEdges:
                if adjacentEdge in visited or adjacentEdge in cantVisit:
                    continue
                possiblePath = False
                # Is None or don't belong to the player?
                if gameState.boardEdges[adjacentEdge].construction is not None and \
                    gameState.boardEdges[adjacentEdge].construction.owner == playerNumber:
                    possiblePath = True

                if possiblePath:
                    # Is adjacent to another player's settlement/city?
                    for node in gameState.boardEdges[adjacentEdge].adjacentNodes:
                        if gameState.boardNodes[node] is not None and \
                            gameState.boardNodes[node].construction is not None and \
                            gameState.boardNodes[node].construction.owner != playerNumber:
                            possiblePath = False

                if possiblePath:
                    nextPath = visited[:]
                    nextCantVisit = []
                    nextCantVisit += gameState.boardEdges[currRoad].adjacentEdges
                    possiblePaths.append(DepthSearch(self.seatNumber, adjacentEdge, nextPath, nextCantVisit))

            if len(possiblePaths) <= 0:
                return visited

            maxPath  = 0
            path     = None
            for p in possiblePaths:
                if len(p) > maxPath:
                    path     = p
                    maxPath  = len(p)
            return path

        results = []
        for startingRoad in startingRoads:
            results.append(DepthSearch(self.seatNumber, startingRoad, [], []))

        if len(results) <= 0:
            return 0

        results = [len(res) for res in results]

        self.roadCount = max(results)

        return self.roadCount

    def DefaultDiscard(self):
        #ROBOT PLAYER DEFAULT DISCARD METHOD
        if sum(self.resources) > 7:
            # SET THE NEW RESOURCES AS UNKNOWN - (WE CANT KNOW WHAT RESOURCES THE ROBOT HAVE DISCARDED)
            self.resources = [0, 0, 0, 0, 0, int(math.ceil(sum(self.resources) / 2.0))]

    @staticmethod
    def GetModelSelectedActions(gameState, player, model=None):

        #shiftedGame = pickle.loads(pickle.dumps(gameState, -1))
        #shiftedGame.ShiftPlayerToFirstSeat(player)

        if model is None:
            model = Player.Model

        possibleActions = []

        if not gameState.setupDone:
            possibleActions = Player.GetAllPossibleActions_Setup(gameState, player)
        elif gameState.currState == "PLAY":
            possibleActions = Player.GetAllPossibleActions_PreDiceRoll(player)
        elif gameState.currState == "PLAY1":
            possibleActions = Player.GetAllPossibleActions_RegularTurns(gameState, player)
        else:
            possibleActions = Player.GetAllPossibleActions_SpecialTurns(gameState, player)

        return player.FilterActionsWithModel(gameState, model, possibleActions, justBest=False)

    @staticmethod
    def GetAllPossibleActions_Setup(gameState, player):

        if gameState.currState == 'START1A':

            if player.firstSettlementBuild:
                return None

            return [BuildSettlementAction(player.seatNumber, setNode, len(player.settlements)) for setNode in
                    gameState.GetPossibleSettlements(player, True)]

        elif gameState.currState == 'START1B':

            if player.firstRoadBuild:
                return None

            possibleRoads = gameState.GetPossibleRoads(player, True)

            return [BuildRoadAction(player.seatNumber, roadEdge, len(player.roads)) for roadEdge in possibleRoads]

        elif gameState.currState == 'START2A':

            if player.secondSettlementBuild:
                return None

            return [BuildSettlementAction(player.seatNumber, setNode, len(player.settlements)) for setNode in
                    gameState.GetPossibleSettlements(player, True)]

        elif gameState.currState == 'START2B':

            if player.secondRoadBuild:
                return None

            possibleRoads = gameState.GetPossibleRoads(player, True)

            return [BuildRoadAction(player.seatNumber, roadEdge, len(player.roads)) for roadEdge in possibleRoads]

    @staticmethod
    def GetAllPossibleActions_PreDiceRoll(player):
        possibleActions = []
        if not player.rolledTheDices and \
                not player.playedDevCard and \
                player.mayPlayDevCards[KNIGHT_CARD_INDEX] and \
                player.developmentCards[KNIGHT_CARD_INDEX] > 0:
            possibleActions.append(UseKnightsCardAction(player.seatNumber, None, None))

        # NOTE: removing if part and just returning fixed an error before
        # if not player.rolledTheDices:
        possibleActions.append(RollDicesAction(player.seatNumber))
        
        return possibleActions

    @staticmethod
    def GetAllPossibleActions_RegularTurns(gameState, player):

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
                    possibleCardsToUse += player.GetMonopolyResource(gameState, player)

                if player.developmentCards[YEAR_OF_PLENTY_CARD_INDEX] > 0 and \
                        player.mayPlayDevCards[YEAR_OF_PLENTY_CARD_INDEX]:
                    possibleCardsToUse += player.GetYearOfPlentyResource(gameState, player)

                if player.developmentCards[ROAD_BUILDING_CARD_INDEX] > 0 and \
                        player.mayPlayDevCards[ROAD_BUILDING_CARD_INDEX] and \
                        player.numberOfPieces[0] > 0:
                    possibleCardsToUse += [UseFreeRoadsCardAction(player.seatNumber, None, None)]

            if possibleCardsToUse:
                possibleActions += possibleCardsToUse

        possibleTrade = player.GetPossibleBankTrades(gameState, player)
        if possibleTrade is not None and possibleTrade:
            possibleActions.append(possibleTrade[int(random.random() * len(possibleTrade))])

        possibleActions.append(EndTurnAction(playerNumber=player.seatNumber))

        return possibleActions

    @staticmethod
    def GetAllPossibleActions_SpecialTurns(gameState, player):

        if gameState.currState == 'PLACING_ROBBER':

            # Rolled out 7  * or *  Used a knight card
            return player.ChooseRobberPosition(gameState, player)

        elif gameState.currState == 'WAITING_FOR_DISCARDS':

            return [player.ChooseCardsToDiscard(player)]

        elif gameState.currState == 'WAITING_FOR_CHOICE':

            return [player.ChoosePlayerToStealFrom(gameState, player)]

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

    # def FilterActionsWithModel(self, gameState, model, actions, justBest = False):

    #     if len(actions) > 1:

    #         actionValues = []

    #         for action in actions:
    #             gameStateCopy = pickle.loads(pickle.dumps(gameState, -1))
    #             action.ApplyAction(gameStateCopy)
    #             dataFrame = GetGameStateDataFrame(gameStateCopy, action, gameState.boardConfig)
    #             score = model.predict(dataFrame)
    #             actionValues.append((score, action))

    #         actionValues = sorted(actionValues, key=lambda actval: actval[0])
    #         result = []
    #         best = -1
    #         worstDelta = actionValues[0][0] - actionValues[-1][0]
    #         for actionVal in actionValues:
    #             if best == -1:
    #                 best = actionVal[0]
    #                 result.append(actionVal[1])
    #             else:
    #                 if justBest:
    #                     if best - actionVal[0] == 0:
    #                         result.append(actionVal[1])
    #                     else:
    #                         break
    #                 else:
    #                     if best - actionVal[0] <= worstDelta * 0.5:
    #                         result.append(actionVal[1])
    #                     else:
    #                         break

    #         return result

    #     else:
    #         return actions

    def GetPossibleActions(self, game, player=None, gameState=None, ignoreTurn=False):
        pass

    def DoMove(self, game):
        pass

    def ChooseCardsToDiscard(self, game, player=None):
        pass

    def ChooseRobberPosition(self, game, player=None):
        pass

    def ChoosePlayerToStealFrom(self, game, player=None):
        pass

    def GetPossibleBankTrades(self, game, player=None):
        pass

    def GetMonopolyResource(self, game, player=None):
        pass

    def GetYearOfPlentyResource(self, game, player=None):
        pass
