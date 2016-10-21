from CatanPlayer import *
import random
import logging
import math

class AgentRandom(Player):

    def __init__(self, name, seatNumber):

        super(AgentRandom, self).__init__(name, seatNumber)

        self.agentName = "RANDOM"

    def GetPossibleActions(self, gameState, player = None):

        if player is None:
            player = self

        if   gameState.currState == 'START1A':

            if player.firstSettlementBuild:
                return None

            possibleSettlements = gameState.GetPossibleSettlements(player, True)

            def RateNode(node, uniqueness):

                possibleResources = [gameState.boardHexes[boardHex].production for boardHex in node.adjacentHexes
                                     if boardHex is not None]

                if len(possibleResources) < 2:
                    return False

                seen   = []
                unique = 0

                for i in range(0, len(possibleResources)):
                    if possibleResources[i] is not None:
                        if possibleResources[i] not in seen:
                            unique += 1
                        seen.append(possibleResources[i])

                if unique < uniqueness:
                    return False

                return True

            for i in range(0, 3):

                goodNodes = [ setNode for setNode in possibleSettlements if RateNode(setNode, 3 - i) ]

                if len(goodNodes) > 0:
                    break

            return [BuildSettlementAction(player.seatNumber, setNode.index, len(player.settlements))
                    for setNode in goodNodes]

        elif gameState.currState == 'START1B':

            if player.firstRoadBuild:
                return None

            possibleRoads = gameState.GetPossibleRoads(player, True)

            return [BuildRoadAction(player.seatNumber, roadEdge.index, len(player.roads)) for roadEdge in possibleRoads]

        elif gameState.currState == 'START2A':

            if player.secondSettlementBuild:
                return None

            possibleSettlements = gameState.GetPossibleSettlements(player, True)

            def RateNode(node, ownedResources, uniqueness):

                possibleResources = [gameState.boardHexes[boardHex].production for boardHex in node.adjacentHexes
                                     if boardHex is not None]

                if len(possibleResources) < 2:
                    return False

                seen = []
                unique = 0

                for i in range(0, len(possibleResources)):
                    if possibleResources[i] is not None:
                        if possibleResources[i] not in seen \
                                and possibleResources[i] not in ownedResources:
                            unique += 1
                        seen.append(possibleResources[i])

                if unique < uniqueness:
                    return False

                return True

            for i in range(0, 3):

                goodNodes = [ setNode for setNode in possibleSettlements if RateNode(setNode, player.resources, 3 - i) ]

                if len(goodNodes) > 0:
                    break

            return [BuildSettlementAction(player.seatNumber, setNode.index, len(player.settlements))
                    for setNode in goodNodes]

        elif gameState.currState == 'START2B':

            if player.secondRoadBuild:
                return None

            possibleRoads = gameState.GetPossibleRoads(player, True)

            return [BuildRoadAction(player.seatNumber, roadEdge.index, len(player.roads))
                    for roadEdge in possibleRoads]

        elif gameState.currState == 'PLAY':

            # FIXME: TRYING TO USE KNIGHT 2 TIMES!!! WHAT??? (fixed?)

            if not player.rolledTheDices and \
               not player.playedDevCard and \
                    player.mayPlayDevCards[KNIGHT_CARD_INDEX] and \
                            player.developmentCards[KNIGHT_CARD_INDEX] > 0:

                return [ UseKnightsCardAction( player.seatNumber, None, None ) ]

            if not player.rolledTheDices:

                return [ RollDicesAction( player.seatNumber ) ]

        elif gameState.currState == 'PLAY1':

            actions             = ['buildRoad', 'buildSettlement', 'buildCity',
                                   'buyDevCard', 'useDevCard']

            possibleActions     = []

            if player.CanAfford(BuildRoadAction.cost) and \
               player.HavePiece(g_pieces.index('ROADS')) > 0:

                possibleActions.append(actions[0])

            if player.CanAfford(BuildSettlementAction.cost) and \
               player.HavePiece(g_pieces.index('SETTLEMENTS')) > 0:

                possibleActions.append(actions[1])

            if player.CanAfford(BuildCityAction.cost) and \
               player.HavePiece(g_pieces.index('CITIES')) > 0 and \
               len(player.settlements) > 0:

                possibleActions.append(actions[2])

            if gameState.CanBuyADevCard(player) and not player.biggestArmy:
                possibleActions.append(actions[3])

            if not player.playedDevCard and sum(player.developmentCards[:-1]) > 0:
                possibleActions.append(actions[4])

            if len(possibleActions) == 0:

                if random.random() >= 0.5:
                    return player.GetPossibleBankTrades(gameState, player)
                else:
                    return None

            chosenAction = random.choice(possibleActions)

            if chosenAction == 'buildRoad':

                possibleRoads = gameState.GetPossibleRoads(player)

                if possibleRoads is not None and len(possibleRoads) > 0:

                    return [BuildRoadAction(player.seatNumber, roadEdge.index, len(player.roads))
                            for roadEdge in possibleRoads]

            elif chosenAction == 'buildSettlement':

                possibleSettlements = gameState.GetPossibleSettlements(player)

                if possibleSettlements is not None and len(possibleSettlements) > 0:

                    return [BuildSettlementAction(player.seatNumber, setNode.index, len(player.settlements))
                            for setNode in possibleSettlements]

            elif chosenAction == 'buildCity':

                possibleCities = gameState.GetPossibleCities(player)

                if possibleCities is not None and len(possibleCities) > 0:

                    return [BuildCityAction(player.seatNumber, setNode.index, len(player.cities))
                            for setNode in possibleCities]

            elif chosenAction == 'buyDevCard':

                return [BuyDevelopmentCardAction(player.seatNumber)]

            elif chosenAction == 'useDevCard':

                possibleCardsToUse = []

                if not player.playedDevCard:

                    if player.developmentCards[MONOPOLY_CARD_INDEX] > 0 and \
                            player.mayPlayDevCards[MONOPOLY_CARD_INDEX]:
                        possibleCardsToUse += player.GetMonopolyResource(player)

                    if player.developmentCards[YEAR_OF_PLENTY_CARD_INDEX] > 0 and \
                            player.mayPlayDevCards[YEAR_OF_PLENTY_CARD_INDEX]:
                        possibleCardsToUse += player.GetYearOfPlentyResource(player)

                    if player.developmentCards[ROAD_BUILDING_CARD_INDEX] > 0 and \
                            player.mayPlayDevCards[ROAD_BUILDING_CARD_INDEX] and \
                                    player.numberOfPieces[0] > 0:
                        possibleCardsToUse += [UseFreeRoadsCardAction(player.seatNumber, None, None)]

                return possibleCardsToUse

        elif gameState.currState == 'PLACING_ROBBER':

            # Rolled out 7  * or *  Used a knight card
            return player.ChooseRobberPosition(gameState, player)

        elif gameState.currState == 'WAITING_FOR_DISCARDS':

            return [player.ChooseCardsToDiscard(player)]

        elif gameState.currState == 'WAITING_FOR_CHOICE':

            return [player.ChoosePlayerToStealFrom(gameState)]

        elif gameState.currState == "PLACING_FREE_ROAD1":

            possibleRoads = gameState.GetPossibleRoads(player, freeRoad=True)

            if possibleRoads is None or len(possibleRoads) <= 0:
                return [ ChangeGameStateAction("PLAY1") ]

            return [BuildRoadAction(player.seatNumber, roadEdge.index,
                                    len(player.roads))
                    for roadEdge in possibleRoads]

        elif gameState.currState == "PLACING_FREE_ROAD2":

            possibleRoads = gameState.GetPossibleRoads(player, freeRoad=True)

            if possibleRoads is None or len(possibleRoads) <= 0:
                return [ ChangeGameStateAction("PLAY1") ]

            return [BuildRoadAction(player.seatNumber, roadEdge.index,
                                    len(player.roads))
                    for roadEdge in possibleRoads]

        return None

    def DoMove(self, game):

        if game.gameState.currPlayer != self.seatNumber and \
            game.gameState.currState != "WAITING_FOR_DISCARDS":
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        logging.debug("possible actions = {0}".format(possibleActions))

        if game.gameState.currState == "PLAY1":

            if possibleActions is not None and len(possibleActions) > 0:
                return random.choice(possibleActions)

            return EndTurnAction(self.seatNumber)

        if possibleActions is not None and len(possibleActions) > 0:
            return random.choice(possibleActions)
        elif possibleActions is None:
            print("NONE!!!")

        return None

    def ChooseCardsToDiscard(self, player = None):

        if player is None:
            player = self

        if sum(player.resources) <= 7:
            return DiscardResourcesAction(player.seatNumber, [0, 0, 0, 0, 0, 0])

        resourcesPopulation = [0 for i in range(0, player.resources[0])] + \
                              [1 for j in range(0, player.resources[1])] + \
                              [2 for k in range(0, player.resources[2])] + \
                              [3 for l in range(0, player.resources[3])] + \
                              [4 for m in range(0, player.resources[4])] + \
                              [5 for n in range(0, player.resources[5])]

        discardCardCount = int(math.floor(len(resourcesPopulation) / 2.0))

        if discardCardCount > 0:
            #assert(player.discardCardCount == discardCardCount, "calculated cards to discard different from server!")
            player.discardCardCount = 0

        selectedResources = random.sample(resourcesPopulation, discardCardCount)

        return DiscardResourcesAction(player.seatNumber, [selectedResources.count(0),
                                                          selectedResources.count(1),
                                                          selectedResources.count(2),
                                                          selectedResources.count(3),
                                                          selectedResources.count(4),
                                                          selectedResources.count(5)])

    def ChooseRobberPosition(self, gameState, player = None):

        possiblePositions = gameState.possibleRobberPos + [gameState.robberPos]

        return [PlaceRobberAction(player.seatNumber, position)
                for position in possiblePositions]

    def ChoosePlayerToStealFrom(self, gameState, player = None):

        if player is None:
            player = self

        possiblePlayers = gameState.GetPossiblePlayersToSteal(player.seatNumber)

        if len(possiblePlayers) > 0:
            return ChoosePlayerToStealFromAction(player.seatNumber, random.choice(possiblePlayers))

        return None

    def GetPossibleBankTrades(self, gameState, player = None):

        if player is None:
            player = self

        availablePorts = self.GetPorts(gameState)

        if availablePorts[-1]:
            minTradeRate = 3
        else:
            minTradeRate = 4

        tradeRates = [minTradeRate, minTradeRate, minTradeRate, minTradeRate, minTradeRate]

        for i in range(0, len(tradeRates)):
            if availablePorts[i]:
                tradeRates[i] = 2

        possibleTradeAmount = [0, 0, 0, 0, 0]
        candidateForTrade   = []

        minResourceAmount = min(player.resources[:-1]) #Don't count the 'UNKNOWN' resource

        for i in range(len(possibleTradeAmount)):
            possibleTradeAmount[i] = int(player.resources[i] / tradeRates[i])
            if player.resources[i] == minResourceAmount:
                candidateForTrade.append(i)

        tradeAmount = random.randint(0, sum(possibleTradeAmount))

        if tradeAmount > 0 and len(candidateForTrade) > 0:

            possibleTradePopulation = [0 for i in range(0, possibleTradeAmount[0])] + \
                                      [1 for j in range(0, possibleTradeAmount[1])] + \
                                      [2 for k in range(0, possibleTradeAmount[2])] + \
                                      [3 for l in range(0, possibleTradeAmount[3])] + \
                                      [4 for m in range(0, possibleTradeAmount[4])]

            logging.debug("Player {0} is checking if he can trade...\n"
                          " He have this resources: {1}\n"
                          " And he thinks he can trade these: {2}".format(player.name, player.resources,
                                                                          possibleTradeAmount))

            chosenResources   = random.sample(possibleTradePopulation, tradeAmount)

            expectedResources = []
            for i in range(0, tradeAmount):
                expectedResources.append(random.choice(candidateForTrade))

            logging.debug("Chosen: {0}\n Expected: {1}\n MaxTrades: {2}".format(
                chosenResources, expectedResources, tradeAmount
            ))

            give = [chosenResources.count(0) * tradeRates[0], chosenResources.count(1) * tradeRates[1],
                    chosenResources.count(2) * tradeRates[2], chosenResources.count(3) * tradeRates[3],
                    chosenResources.count(4) * tradeRates[4]]

            get  = [expectedResources.count(0), expectedResources.count(1),
                    expectedResources.count(2), expectedResources.count(3),
                    expectedResources.count(4)]

            logging.debug("Player {0} will trade with the bank!\n"
                          " GIVE = {1}\n"
                          " GET  = {2}".format(player.name, give, get))

            return [ BankTradeOfferAction(player.seatNumber, give, get) ]

        return None

    def GetMonopolyResource(self, game, player = None):

        if player is None:
            player = self

        candidateResource = []

        minResourceAmount = min(player.resources[:-1])

        for i in range(0, len(player.resources) - 1):

            if player.resources[i] == minResourceAmount:
                candidateResource.append(i + 1)

        if len(candidateResource) <= 0:

            randomPick = random.choice([1,2,3,4,5])

            logging.critical("Monopoly pick FAILED!!!! Picking at random: {0}".format(randomPick))

            chosenResource = randomPick

        else:

            chosenResource = random.choice(candidateResource)

        return [ UseMonopolyCardAction(player.seatNumber, chosenResource) ]

    def GetYearOfPlentyResource(self, game, player = None):

        if player is None:
            player = self

        candidateResource = []

        chosenResources = [0, 0, 0, 0, 0]

        minResourceAmount = min(player.resources[:-1])

        for i in range(0, len(player.resources) - 1):

            if player.resources[i] == minResourceAmount:
                candidateResource.append(i)

        if len(candidateResource) == 1:

            chosenResources[i] = 2

        else:

            pick1 = random.choice(candidateResource)
            pick2 = random.choice(candidateResource)

            chosenResources[pick1] += 1
            chosenResources[pick2] += 1

        return [ UseYearOfPlentyCardAction(player.seatNumber, chosenResources) ]

    def UpdateResourcesFromServer(self, action, element, value):

        if element in g_resources:  # RESOURCE

            if action == 'SET':
                self.resources[g_resources.index(element)] = value

            elif action == 'GAIN':
                self.resources[g_resources.index(element)] += value

            elif action == 'LOSE':
                self.resources[g_resources.index(element)] -= value

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