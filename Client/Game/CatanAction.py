from JSettlersMessages import *
from Game.CatanBoard import *

from Game.CatanUtilsPy import listm

import random

freeBuildStates = ["START1A", "START1B", "START2A", "START2B",
                   "PLACING_ROAD", "PLACING_SETTLEMENT", "PLACING_CITY",
                   "PLACING_FREE_ROAD1", "PLACING_FREE_ROAD2"]

putPieceStates = ["START1A", "START1B", "START2A", "START2B",
                  "PLACING_ROAD", "PLACING_SETTLEMENT", "PLACING_CITY",
                  "PLACING_FREE_ROAD1", "PLACING_FREE_ROAD2"]

class Action(object):

    def __init__(self):

        self.tradeOptimistic = False
        pass

    def GetMessage(self, gameName, currGameStateName = None):
        pass

    def ApplyAction(self, gameState):
        pass

    def __eq__(self, other):
        if other is None:
            if self is None:
                return True
            return False
        return self.__dict__ == other.__dict__
    
    def getString(self):
        pass

class BuildAction(Action):

    def __init__(self, playerNumber, position, index, pieceId, cost):

        super(BuildAction, self).__init__()
        self.playerNumber = playerNumber
        self.position     = position
        self.index        = index
        self.pieceId      = pieceId
        self.cost         = cost

    def GetMessage(self, gameName, currGameStateName = None):

        if currGameStateName is not None and \
            currGameStateName in putPieceStates:

            return PutPieceMessage(gameName, self.playerNumber, self.pieceId, self.position)

        return BuildRequestMessage(gameName, self.pieceId)


    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = BuildAction")

        gameState.players[self.playerNumber].Build(gameState, g_constructionTypes[self.pieceId][0], self.position)

        if gameState.currState not in freeBuildStates:

            gameState.players[self.playerNumber].resources -= self.cost

            if self.tradeOptimistic:
                discountCount = 0
                for i in range(0, len(gameState.players[self.playerNumber].resources)):
                    if gameState.players[self.playerNumber].resources[i] < 0:
                        discountCount += gameState.players[self.playerNumber].resources[i] * -1
                        gameState.players[self.playerNumber].resources[i] = 0
                gameState.players[self.playerNumber].DiscountAtRandom(discountCount)

    def __str__(self):

        return "Build {0} Action:  \n" \
               "    player   = {1} \n" \
               "    position = {2} \n" \
               "    index    = {3}".format(
            g_constructionTypes[self.pieceId][0],
            self.playerNumber,
            hex(self.position),
            self.index
        )

class BuildRoadAction(BuildAction):

    type = 'BuildRoad'
    cost = listm([ 1,  # brick
                   0,  # ore
                   0,  # wool
                   0,  # grain
                   1,  # lumber
                   0 ]) # unknown

    pieceId = 0

    def __init__(self, playerNumber, position, index):
        super(BuildRoadAction, self).__init__(playerNumber, position, index,
                                              BuildRoadAction.pieceId, BuildRoadAction.cost)

    def ApplyAction(self, gameState):

        super(BuildRoadAction, self).ApplyAction(gameState)

        # If the player has atleast 5 roads update
        if len(gameState.players[self.playerNumber].roads) >= 5:#gameState.checkLongestRoad:
            gameState.UpdateLongestRoad()

        if gameState.currState == "START1B":

            gameState.players[gameState.currPlayer].firstRoadBuild = True

            nextPlayer = (gameState.currPlayer + 1) % len(gameState.players)

            if nextPlayer == gameState.startingPlayer:
                gameState.currState = "START2A"

            else:
                gameState.currPlayer = nextPlayer
                gameState.currState = "START1A"

        elif gameState.currState == "START2B":

            gameState.players[gameState.currPlayer].secondRoadBuild = True

            if gameState.currPlayer == gameState.startingPlayer:
                gameState.FinishSetup()
                gameState.currState = "PLAY"
            else:
                nextPlayer = (gameState.currPlayer - 1) % len(gameState.players)
                gameState.currPlayer = nextPlayer
                gameState.currState = "START2A"

        elif gameState.currState == "PLACING_FREE_ROAD1":
            gameState.currState = "PLACING_FREE_ROAD2"

        elif gameState.currState == "PLACING_FREE_ROAD2":
            gameState.currState = "PLAY1"
    
    def getString(self):
        return f"{self.type}{self.position}"


class BuildSettlementAction(BuildAction):

    type = 'BuildSettlement'
    cost = listm([ 1,  # brick
                   0,  # ore
                   1,  # wool
                   1,  # grain
                   1,  # lumber
                   0 ]) # unknown

    pieceId = 1

    def __init__(self, playerNumber, position, index):
        super(BuildSettlementAction, self).__init__(playerNumber, position, index,
                                                    BuildSettlementAction.pieceId, BuildSettlementAction.cost)

    def ApplyAction(self, gameState):

        super(BuildSettlementAction, self).ApplyAction(gameState)

        #if gameState.boardNodes[self.position].portType is not None:
        #    gameState.players[self.playerNumber].UpdateTradeRates(gameState)

        if gameState.checkLongestRoad:

            for edge in gameState.boardNodes[self.position].adjacentEdges:
                checkLongestRoad = False
                if gameState.boardEdges[edge].construction is not None and \
                   gameState.boardEdges[edge].construction.owner != self.playerNumber:
                    checkLongestRoad = True
                    break

            if checkLongestRoad:
                gameState.UpdateLongestRoad()

        if gameState.currState == "START1A":

            gameState.players[gameState.currPlayer].firstSettlementBuild = True

            gameState.currState = "START1B"

        elif gameState.currState == "START2A":

            gameState.players[gameState.currPlayer].secondSettlementBuild = True

            gameState.players[self.playerNumber].GetStartingResources(gameState)

            gameState.currState = "START2B"

            # Record resource production and trade rates after setup phase
            if gameState.players[self.playerNumber].recordStats:
                for diceNumber, resourceList in gameState.players[self.playerNumber].diceProduction.items():
                    gameState.players[self.playerNumber].stats.setupResourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
                gameState.players[self.playerNumber].stats.setupTradeRates = gameState.players[self.playerNumber].tradeRates
    
    def getString(self):
        return f"{self.type}{self.position}"


class BuildCityAction(BuildAction):

    type = 'BuildCity'
    cost = listm([ 0,  # brick
                   3,  # ore
                   0,  # wool
                   2,  # grain
                   0,  # lumber
                   0 ]) # unknown

    pieceId = 2

    def __init__(self, playerNumber, position, index):
        super(BuildCityAction, self).__init__(playerNumber, position, index,
                                              BuildCityAction.pieceId, BuildCityAction.cost)

    '''
    def ApplyAction(self, gameState):

        super(BuildCityAction, self).ApplyAction(gameState)
    '''

    def getString(self):
        return f"{self.type}{self.position}"

class RollDicesAction(Action):

    type = 'RollDices'

    def __init__(self, playerNumber = None, result = None):

        self.playerNumber = playerNumber

        if result is not None:
            self.result = result
        else:
            self.result = 2 + int(random.random() * 6) + int(random.random() * 6)

    def GetMessage(self, gameName, currGameStateName = None):

        return RollDiceMessage(gameName)

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(RollDicesAction.type))

        gameState.players[self.playerNumber].rolledTheDices = True
        gameState.dicesAreRolled = True

        if self.result == 7:

            discardRound = False

            for index in range(0, len(gameState.players)):

                if sum(gameState.players[index].resources) > 7:
                    discardRound = True

            if discardRound:

                gameState.currState = "WAITING_FOR_DISCARDS"

                gameState.playerBeforeDiscards = gameState.currPlayer

                gameState.currPlayer = 0

            else:

                gameState.currState = "PLACING_ROBBER"

        else:
            for player in gameState.players:
                player.UpdatePlayerResources(gameState, self.result)

            gameState.currState = "PLAY1"

    def getString(self):
        return f"{self.type}"

class BuyDevelopmentCardAction(Action):

    type = 'BuyDevelopmentCard'
    cost = listm([ 0,  # brick
                   1,  # ore
                   1,  # wool
                   1,  # grain
                   0,  # lumber
                   0 ]) # unknown

    def __init__(self, playerNumber):

        super(BuyDevelopmentCardAction, self).__init__()
        self.playerNumber = playerNumber

    def GetMessage(self, gameName, currGameStateName = None):

        return BuyCardRequestMessage(gameName)

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(BuyDevelopmentCardAction.type))

        gameState.players[self.playerNumber].resources -= BuyDevelopmentCardAction.cost

        if gameState.players[self.playerNumber].recordStats:
            gameState.players[self.playerNumber].stats.devCardsBought += 1

        if self.tradeOptimistic:
            discountCount = 0
            for i in range(0, len(gameState.players[self.playerNumber].resources)):
                if gameState.players[self.playerNumber].resources[i] < 0:
                    discountCount += gameState.players[self.playerNumber].resources[i] * -1
                    gameState.players[self.playerNumber].resources[i] = 0
            self.players[self.playerNumber].DiscountAtRandom(discountCount)

        gameState.DrawDevCard(self.playerNumber)
    
    def getString(self):
        return f"{self.type}"

class UseDevelopmentCardAction(Action):

    def __init__(self, playerNumber, index):

        self.playerNumber = playerNumber

        self.index        = index

    def ApplyAction(self, gameState):

        gameState.players[self.playerNumber].developmentCards[self.index] -= 1

        gameState.players[self.playerNumber].mayPlayDevCards[self.index] = False

        gameState.players[self.playerNumber].playedDevCard = True

        if gameState.players[self.playerNumber].recordStats:
            gameState.players[self.playerNumber].stats.usedDevCards[self.index] += 1

class UseKnightsCardAction(UseDevelopmentCardAction):

    type = 'UseKnightsCard'

    def __init__(self, playerNumber, newRobberPos, targetPlayerIndex):

        super(UseKnightsCardAction, self).__init__(playerNumber, KNIGHT_CARD_INDEX)
        self.playerNumber      = playerNumber
        self.robberPos         = newRobberPos
        self.targetPlayerIndex = targetPlayerIndex

    def GetMessage(self, gameName, currGameStateName = None):

        return PlayDevCardRequestMessage(gameName, KNIGHT_CARD_INDEX)

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(UseKnightsCardAction.type))

        super(UseKnightsCardAction, self).ApplyAction(gameState)

        gameState.currState = "PLACING_ROBBER"

        gameState.players[self.playerNumber].knights += 1

        gameState.UpdateLargestArmy()
    
    def getString(self):
        return f"{self.type}"

class UseMonopolyCardAction(UseDevelopmentCardAction):

    type = 'UseMonopolyCard'

    def __init__(self, playerNumber, resource):
        if resource > 4:
            raise ValueError("RESOURCE INDEX FOR MONOPOLY SHOULD BE BETWEEN 0-4")

        super(UseMonopolyCardAction, self).__init__(playerNumber, MONOPOLY_CARD_INDEX)

        self.resource     = resource

    def GetMessage(self, gameName, currGameStateName = None):

        return [ PlayDevCardRequestMessage(gameName, self.index),
                 MonopolyPickMessage(gameName, self.resource)   ]

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(UseMonopolyCardAction.type))

        super(UseMonopolyCardAction, self).ApplyAction(gameState)

        total = 0

        for index in range(0, len(gameState.players)):

            if index == self.playerNumber:
                continue

            amount = gameState.players[index].resources[self.resource]

            gameState.players[index].resources[self.resource] = 0

            total += amount
            if gameState.players[self.playerNumber].recordStats:
                gameState.players[index].stats.totalResourcesStolen += amount

        gameState.players[self.playerNumber].resources[self.resource] += total

        if gameState.players[self.playerNumber].recordStats:
            gameState.players[self.playerNumber].stats.resourcesFromDevCard[self.resource] += total
    
    def getString(self):
        return f"{self.type}{self.resource}"

class UseYearOfPlentyCardAction(UseDevelopmentCardAction):

    type = 'UseYearOfPlentyCard'

    def __init__(self, playerNumber, resources):

        super(UseYearOfPlentyCardAction, self).__init__(playerNumber, YEAR_OF_PLENTY_CARD_INDEX)

        self.resources    = resources

    def GetMessage(self, gameName, currGameStateName = None):

        return [PlayDevCardRequestMessage(gameName, self.index),
                DiscoveryPickMessage(gameName, self.resources)                ]

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(UseYearOfPlentyCardAction.type))

        super(UseYearOfPlentyCardAction, self).ApplyAction(gameState)

        gameState.players[self.playerNumber].resources[self.resources[0]] += 1

        gameState.players[self.playerNumber].resources[self.resources[1]] += 1

        if gameState.players[self.playerNumber].recordStats:
            gameState.players[self.playerNumber].stats.resourcesFromDevCard[self.resources[0]] += 1
            gameState.players[self.playerNumber].stats.resourcesFromDevCard[self.resources[1]] += 1
    
    def getString(self):
        return f"{self.type}{self.resources[0]}{self.resources[1]}{self.resources[2]}{self.resources[3]}{self.resources[4]}"

class UseFreeRoadsCardAction(UseDevelopmentCardAction):

    type = 'UseFreeRoadsCard'

    def __init__(self, playerNumber, road1Edge, road2Edge):

        super(UseFreeRoadsCardAction, self).__init__(playerNumber, ROAD_BUILDING_CARD_INDEX)

        self.road1Edge    = road1Edge
        self.road2Edge    = road2Edge

    def GetMessage(self, gameName, currGameStateName = None):

        return PlayDevCardRequestMessage(gameName, self.index)

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(UseFreeRoadsCardAction.type))

        super(UseFreeRoadsCardAction, self).ApplyAction(gameState)

        gameState.currState = "PLACING_FREE_ROAD1"
    
    def getString(self):
        return f"{self.type}"
    

class PlaceRobberAction(Action):

    type = 'PlaceRobber'

    def __init__(self, playerNumber, newRobberPos):

        self.playerNumber = playerNumber
        self.robberPos    = newRobberPos

    def GetMessage(self, gameName, currGameStateName = None):

        return MoveRobberMessage(gameName, self.playerNumber, self.robberPos)

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(PlaceRobberAction.type))

        pastRobberPos = gameState.robberPos

        gameState.players[self.playerNumber].PlaceRobber(gameState, self.robberPos)

        gameState.UpdateRobDiceProduction(gameState, pastRobberPos=pastRobberPos, newRobberPos=gameState.robberPos)

        possiblePlayers = gameState.GetPossiblePlayersToSteal(self.playerNumber)

        if len(possiblePlayers) > 1:
            gameState.currState = "WAITING_FOR_CHOICE"
        else:
            if len(possiblePlayers) == 1:
                ChoosePlayerToStealFromAction(self.playerNumber, possiblePlayers[0]).ApplyAction(gameState)
            else:
                if gameState.dicesAreRolled:
                    gameState.currState = "PLAY1"
                else:
                    gameState.currState = "PLAY"

    def getString(self):
        return f"{self.type}{self.robberPos}"

class EndTurnAction(Action):

    type = 'EndTurn'

    def __init__(self, playerNumber):

        self.playerNumber = playerNumber

    def GetMessage(self, gameName, currGameStateName = None):

        return EndTurnMessage(gameName)

    def ApplyAction(self, gameState):

        gameState.dicesAreRolled = False
        gameState.players[self.playerNumber].rolledTheDices = False
        gameState.players[self.playerNumber].placedRobber   = False
        gameState.currTurn += 1
        if gameState.players[self.playerNumber].recordStats:
            gameState.players[self.playerNumber].stats.numTurns += 1

        playerPoints = gameState.players[self.playerNumber].GetVictoryPoints()

        if playerPoints >= 8 and not gameState.checkLongestRoad:
            gameState.UpdateLongestRoad()
            gameState.checkLongestRoad = True

        if playerPoints >= 10:
            gameState.currState = "OVER"
            gameState.winner    = self.playerNumber

        else:
            gameState.currState = "PLAY"

            gameState.currPlayer = (gameState.currPlayer + 1) % len(gameState.players)

            gameState.players[gameState.currPlayer].UpdateMayPlayDevCards(canUseAll=True)
            gameState.players[gameState.currPlayer].playedDevCard = False

    def getString(self):
        return f"{self.type}"

class DiscardResourcesAction(Action):

    type = 'DiscardResources'

    def __init__(self, playerNumber, resources):

        self.playerNumber = playerNumber
        self.resources    = resources

    def GetMessage(self, gameName, currGameStateName = None):

        return DiscardMessage(gameName, self.resources[0], self.resources[1],
                                        self.resources[2], self.resources[3],
                                        self.resources[4], self.resources[5])

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(DiscardResourcesAction.type))

        gameState.players[self.playerNumber].resources -= self.resources

        if gameState.players[self.playerNumber].recordStats:
            gameState.players[self.playerNumber].stats.totalResourcesDiscarded += sum(self.resources)

        gameState.currPlayer += 1

        if gameState.currPlayer >= len(gameState.players):

            gameState.currPlayer = gameState.playerBeforeDiscards

            gameState.playerBeforeDiscards = -1

            gameState.currState = "PLACING_ROBBER"

    def getString(self):
        # Note self.resources is the discard resources
        if sum(self.resources) <= 5:
            return f"{self.type}{self.resources[0]}{self.resources[1]}{self.resources[2]}{self.resources[3]}{self.resources[4]}{self.resources[5]}"
        else:
            return f"{self.type}>5"

class ChoosePlayerToStealFromAction(Action):

    type = 'ChoosePlayerToStealFrom'

    def __init__(self, playerNumber, targetPlayerNumber):

        self.playerNumber       = playerNumber
        self.targetPlayerNumber = targetPlayerNumber

    def GetMessage(self, gameName, currGameStateName = None):

        return ChoosePlayerMessage(gameName, self.targetPlayerNumber)

    def ApplyAction(self, gameState):

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(ChoosePlayerToStealFromAction.type))

        targetPlayer = gameState.players[self.targetPlayerNumber]

        resourcesPopulation = [0 for i in range(0, targetPlayer.resources[0])] + \
                              [1 for j in range(0, targetPlayer.resources[1])] + \
                              [2 for k in range(0, targetPlayer.resources[2])] + \
                              [3 for l in range(0, targetPlayer.resources[3])] + \
                              [4 for m in range(0, targetPlayer.resources[4])] + \
                              [5 for n in range(0, targetPlayer.resources[5])]

        if len(resourcesPopulation) > 0:

            stolenResource = random.choice(resourcesPopulation)

            gameState.players[self.playerNumber].resources[stolenResource] += 1

            gameState.players[self.targetPlayerNumber].resources[stolenResource] -= 1

            if gameState.players[self.playerNumber].recordStats:
                gameState.players[self.targetPlayerNumber].stats.totalResourcesStolen += 1

        if gameState.dicesAreRolled:
            gameState.currState = "PLAY1"
        else:
            gameState.currState = "PLAY"
        
    def getString(self):
        return f"{self.type}{self.targetPlayerNumber}"

class MakeTradeOfferAction(Action):

    type = 'MakeTradeOffer'

    def __init__(self, fromPlayerNumber, toPlayers, giveResources, getResources):

        self.fromPlayerNumber = fromPlayerNumber

        self.toPlayers                   = toPlayers
        self.toPlayers[fromPlayerNumber] = False #Assert the player cannot offer to himself!

        assert(any(toPlayers) == True)

        self.toPlayerNumbers = []
        for i in range(0, len(self.toPlayers)):
            if self.toPlayers[i]:
                self.toPlayerNumbers.append(i)

        self.giveResources     = giveResources
        self.getResources      = getResources
        self.previousGameState = None

    def GetMessage(self, gameName, currGameStateName = None):

        return MakeOfferMessage(gameName, self.fromPlayerNumber, self.toPlayers, self.giveResources, self.getResources)

    def ApplyAction(self, gameState):

        if gameState.currState != 'WAITING_FOR_TRADE':
            self.previousGameState = gameState.currState
            gameState.currState    = 'WAITING_FOR_TRADE'
        else:
            self.previousGameState = gameState.currTradeOffer.previousGameState
        
        gameState.currPlayer = self.toPlayerNumbers[int(random.random() * len(self.toPlayerNumbers))]

        self.toPlayerNumbers.remove(gameState.currPlayer)
        gameState.currTradeOffer = self

        if gameState.players[self.fromPlayerNumber].recordStats:
            gameState.players[self.fromPlayerNumber].stats.totalPlayerTrades += 1

    def getString(self):
        return f"BankTradeOffer{self.giveResources[0]}{self.giveResources[1]}{self.giveResources[2]}{self.giveResources[3]}{self.giveResources[4]}_{self.getResources[0]}{self.getResources[1]}{self.getResources[2]}{self.getResources[3]}{self.getResources[4]}"


class RejectTradeOfferAction(Action):

    type = 'RejectTradeOffer'

    def __init__(self, playerNumber):

        self.playerNumber = playerNumber

    def GetMessage(self, gameName, currGameStateName = None):

        return RejectOfferMessage(gameName, self.playerNumber)

    def ApplyAction(self, gameState):
        player = gameState.players[self.playerNumber]
        if player.recordStats:
            player.stats.rejectedTrades += 1

        if len(gameState.currTradeOffer.toPlayerNumbers) <= 0:
            gameState.currState      = gameState.currTradeOffer.previousGameState
            gameState.currPlayer     = gameState.currTradeOffer.fromPlayerNumber
            gameState.currTradeOffer = None
        else:
            currPlayerIndex = int(random.random() * len(gameState.currTradeOffer.toPlayerNumbers))
            gameState.currPlayer = gameState.currTradeOffer.toPlayerNumbers[currPlayerIndex]
            gameState.currTradeOffer.toPlayerNumbers.remove(gameState.currPlayer)
    
    # Rejecting incoming offer is treated as EndTurn action
    def getString(self):
        return f"EndTurn"

class AcceptTradeOfferAction(Action):

    type = 'AcceptTradeOffer'

    def __init__(self, playerNumber, offerPlayerNumber, gameState=None):

        self.playerNumber      = playerNumber
        self.offerPlayerNumber = offerPlayerNumber

        if gameState:
            self.iGive = gameState.currTradeOffer.getResources
            self.iGet  = gameState.currTradeOffer.giveResources

    def GetMessage(self, gameName, currGameStateName = None):

        return AcceptOfferMessage(gameName, self.playerNumber, self.offerPlayerNumber)

    def ApplyAction(self, gameState):
        player = gameState.players[self.playerNumber]
        offerPlayer = gameState.players[self.offerPlayerNumber]

        # Collect pre trade info
        if player.recordStats:
            possibleSettlementsBefore = gameState.GetPossibleSettlements(player)
            canBuildSettlementBefore = possibleSettlementsBefore and player.HavePiece(g_pieces.index('SETTLEMENTS')) and player.CanAfford(BuildSettlementAction.cost)
            canBuildCityBefore = player.settlements and player.CanAfford(BuildCityAction.cost)
            canBuyDevCardBefore = player.CanAfford(BuyDevelopmentCardAction.cost)
            canBuildRoadBefore = gameState.GetPossibleRoads(player) and player.HavePiece(g_pieces.index('ROADS')) and player.CanAfford(BuildRoadAction.cost)
        if offerPlayer.recordStats:
            possibleSettlementsBeforeOffer = gameState.GetPossibleSettlements(offerPlayer)
            canBuildSettlementBeforeOffer = possibleSettlementsBeforeOffer and offerPlayer.HavePiece(g_pieces.index('SETTLEMENTS')) and offerPlayer.CanAfford(BuildSettlementAction.cost)
            canBuildCityBeforeOffer = offerPlayer.settlements and offerPlayer.CanAfford(BuildCityAction.cost)
            canBuyDevCardBeforeOffer = offerPlayer.CanAfford(BuyDevelopmentCardAction.cost)
            canBuildRoadBeforeOffer = gameState.GetPossibleRoads(offerPlayer) and offerPlayer.HavePiece(g_pieces.index('ROADS')) and offerPlayer.CanAfford(BuildRoadAction.cost)

        gameState.currState  = gameState.currTradeOffer.previousGameState
        gameState.currPlayer = gameState.currTradeOffer.fromPlayerNumber

        give = gameState.currTradeOffer.giveResources
        get  = gameState.currTradeOffer.getResources

        # Presuming offerPlayer is the player who made the offer, and playerNumber is accepting offer
        gameState.players[self.offerPlayerNumber].resources -= listm(give)
        gameState.players[self.offerPlayerNumber].resources += listm(get)

        gameState.players[self.playerNumber].resources      -= listm(get)
        gameState.players[self.playerNumber].resources      += listm(give)

        gameState.currTradeOffer = None

        # Add Incoming trade stats
        if player.recordStats:
            player.stats.acceptedTrades += 1
            canBuildSettlementAfter = player.CanAfford(BuildSettlementAction.cost)
            canBuildRoadAfter = player.CanAfford(BuildRoadAction.cost)
            canBuildCityAfter = player.CanAfford(BuildCityAction.cost)
            canBuyDevCardAfter = player.CanAfford(BuyDevelopmentCardAction.cost)
                # Trades which allow us to build
            if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                player.stats.goodSettlementAcceptedTrades += 1
            if canBuildCityBefore == False and canBuildCityAfter == True:
                player.stats.goodCityAcceptedTrades += 1
            if canBuildRoadBefore == False and canBuildRoadAfter == True :
                player.stats.goodRoadAcceptedTrades += 1
            if canBuyDevCardBefore == False and canBuyDevCardAfter == True:
                player.stats.goodDevCardAcceptedTrades += 1
                # Trades which get rid of resources for possible Builds
            if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                player.stats.badSettlementAcceptedTrades += 1
            if canBuildCityBefore == True and canBuildCityAfter == False:
                player.stats.badCityAcceptedTrades += 1
            if canBuyDevCardBefore == True and canBuyDevCardAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuildRoadAfter == False:
                player.stats.badDevCardAcceptedTrades += 1
            if canBuildRoadBefore == True and canBuildRoadAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuyDevCardAfter == False:
                player.stats.badRoadAcceptedTrades += 1
            if canBuildSettlementAfter == False and canBuildRoadAfter == False and canBuildCityAfter == False and canBuyDevCardAfter == False:
                player.stats.neutralAcceptedTrades += 1
        
        # Add Offering trade stats
        if offerPlayer.recordStats:
            offerPlayer.stats.acceptedPlayerTrades += 1
            canBuildSettlementAfterOffer = offerPlayer.CanAfford(BuildSettlementAction.cost)
            canBuildRoadAfterOffer = offerPlayer.CanAfford(BuildRoadAction.cost)
            canBuildCityAfterOffer = offerPlayer.CanAfford(BuildCityAction.cost)
            canBuyDevCardAfterOffer = offerPlayer.CanAfford(BuyDevelopmentCardAction.cost)
                # Trades which allow us to build
            if canBuildSettlementBeforeOffer == False and canBuildSettlementAfterOffer == True:
                offerPlayer.stats.goodSettlementPlayerTrades += 1
            if canBuildCityBeforeOffer == False and canBuildCityAfterOffer == True:
                offerPlayer.stats.goodCityPlayerTrades += 1
            if canBuildRoadBeforeOffer == False and canBuildRoadAfterOffer == True :
                offerPlayer.stats.goodRoadPlayerTrades += 1
            if canBuyDevCardBeforeOffer == False and canBuyDevCardAfterOffer == True:
                offerPlayer.stats.goodDevCardPlayerTrades += 1
                # Trades which get rid of resources for possible Builds
            if canBuildSettlementBeforeOffer == True and canBuildSettlementAfterOffer == False:
                offerPlayer.stats.badSettlementPlayerTrades += 1
            if canBuildCityBeforeOffer == True and canBuildCityAfterOffer == False:
                offerPlayer.stats.badCityPlayerTrades += 1
            if canBuyDevCardBeforeOffer == True and canBuyDevCardAfterOffer == False and canBuildCityAfterOffer == False and canBuildSettlementAfterOffer == False and canBuildRoadAfterOffer == False:
                offerPlayer.stats.badDevCardPlayerTrades += 1
            if canBuildRoadBeforeOffer == True and canBuildRoadAfterOffer == False and canBuildCityAfterOffer == False and canBuildSettlementAfterOffer == False and canBuyDevCardAfterOffer == False:
                offerPlayer.stats.badRoadPlayerTrades += 1
            if canBuildSettlementAfterOffer == False and canBuildRoadAfterOffer == False and canBuildCityAfterOffer == False and canBuyDevCardAfterOffer == False:
                offerPlayer.stats.neutralPlayerTrades += 1
    
    # Accepting offer is treated as BankTradeOffer
    def getString(self):
        return f"BankTradeOffer{self.iGive[0]}{self.iGive[1]}{self.iGive[2]}{self.iGive[3]}{self.iGive[4]}_{self.iGet[0]}{self.iGet[1]}{self.iGet[2]}{self.iGet[3]}{self.iGet[4]}"

class BankTradeOfferAction(Action):

    type = 'BankTradeOffer'

    def __init__(self, playerNumber, giveResources, getResources):

        self.playerNumber  = playerNumber
        self.giveResources = giveResources
        self.getResources  = getResources

    def GetMessage(self, gameName, currGameStateName = None):

        return BankTradeMessage(gameName, self.giveResources, self.getResources)

    def ApplyAction(self, gameState):
        player = gameState.players[self.playerNumber]

        # Collect pre trade info
        if player.recordStats:
            possibleSettlementsBefore = gameState.GetPossibleSettlements(player)
            canBuildSettlementBefore = possibleSettlementsBefore and player.HavePiece(g_pieces.index('SETTLEMENTS')) and player.CanAfford(BuildSettlementAction.cost)
            canBuildCityBefore = player.settlements and player.CanAfford(BuildCityAction.cost)
            canBuyDevCardBefore = player.CanAfford(BuyDevelopmentCardAction.cost)
            canBuildRoadBefore = gameState.GetPossibleRoads(player) and player.HavePiece(g_pieces.index('ROADS')) and player.CanAfford(BuildRoadAction.cost)

        #logging.debug("APPLYING ACTION! \n TYPE = {0}".format(BankTradeOfferAction.type))

        # ADD THE 'UNKNOWN' RESOURCE TYPE (not present in trade transaction)
        give = self.giveResources + [0]
        get  = self.getResources  + [0]

        player.resources -= listm(give)
        player.resources += listm(get )

        # Add stats
        if player.recordStats:
            canBuildSettlementAfter = player.CanAfford(BuildSettlementAction.cost)
            canBuildRoadAfter = player.CanAfford(BuildRoadAction.cost)
            canBuildCityAfter = player.CanAfford(BuildCityAction.cost)
            canBuyDevCardAfter = player.CanAfford(BuyDevelopmentCardAction.cost)
            # Trades which allow us to build
            if canBuildSettlementBefore == False and canBuildSettlementAfter == True:
                player.stats.goodSettlementBankTrades += 1
            if canBuildCityBefore == False and canBuildCityAfter == True:
                player.stats.goodCityBankTrades += 1
            if canBuildRoadBefore == False and canBuildRoadAfter == True :
                player.stats.goodRoadBankTrades += 1
            if canBuyDevCardBefore == False and canBuyDevCardAfter == True:
                player.stats.goodDevCardBankTrades += 1
            # Trades which get rid of resources for possible Builds
            if canBuildSettlementBefore == True and canBuildSettlementAfter == False:
                player.stats.badSettlementBankTrades += 1
            if canBuildCityBefore == True and canBuildCityAfter == False:
                player.stats.badCityBankTrades += 1
            if canBuyDevCardBefore == True and canBuyDevCardAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuildRoadAfter == False:
                player.stats.badDevCardBankTrades += 1
            if canBuildRoadBefore == True and canBuildRoadAfter == False and canBuildCityAfter == False and canBuildSettlementAfter == False and canBuyDevCardAfter == False:
                player.stats.badRoadBankTrades += 1
            if canBuildSettlementAfter == False and canBuildRoadAfter == False and canBuildCityAfter == False and canBuyDevCardAfter == False:
                player.stats.neutralBankTrades += 1

            gameState.players[self.playerNumber].stats.resourcesFromBankTrade += listm(get)
    
    def getString(self):
        return f"{self.type}{self.giveResources[0]}{self.giveResources[1]}{self.giveResources[2]}{self.giveResources[3]}{self.giveResources[4]}_{self.getResources[0]}{self.getResources[1]}{self.getResources[2]}{self.getResources[3]}{self.getResources[4]}"

class ChangeGameStateAction(Action):

    type = 'ChangeGameState'

    def __init__(self, newGameState):
        self.gameState = newGameState

    def GetMessage(self, gameName, currGameStateName=None):
        return None

    def ApplyAction(self, gameState):

        gameState.currState = self.gameState