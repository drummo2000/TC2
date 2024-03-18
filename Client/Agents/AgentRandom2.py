from Game.CatanGame import GameState
from Game.CatanPlayer import Player
from Game.CatanAction import *
from itertools import combinations
import math
import random

class AgentRandom2(Player):

    def __init__(self, name, seatNumber, recordStats=False, playerTrading: bool=False):

        super(AgentRandom2, self).__init__(name, seatNumber, recordStats=recordStats)
        self.playerTrading          = playerTrading

        self.trading                = None
    
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

        possibleBankTrades = self.GetPossibleBankTrades(gameState, player)
        if possibleBankTrades is not None and possibleBankTrades:
            possibleActions += possibleBankTrades
        
        # Can only offer 2 player trades per turn
        if self.playerTrading:
            if self.tradeCount < 2:
                possiblePlayerTrades = self.GetPossiblePlayerTrades(gameState, player)
                if possiblePlayerTrades is not None and possiblePlayerTrades:
                    possibleActions += possiblePlayerTrades

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
                action = [ ChangeGameStateAction("PLAY1") ]
            else:
                action = [BuildRoadAction(player.seatNumber, roadEdge,
                                        len(player.roads))
                        for roadEdge in possibleRoads]
            return action

        elif gameState.currState == "WAITING_FOR_TRADE":

            return self.GetPossiblePlayerTradeReactions(gameState, player)

    # Returns list of possible actions in given state
    def GetPossibleActions(self, gameState, player = None):

        if player is None:
            player = self
        
        # Call function based on game phase
        if not gameState.setupDone:
            actions = self.GetAllPossibleActions_Setup(gameState, player)
        elif gameState.currState == "PLAY":
            self.tradeCount = 0
            actions = self.GetAllPossibleActions_PreDiceRoll(player)
        elif gameState.currState == "PLAY1":
            actions = self.GetAllPossibleActions_RegularTurns(gameState, player)
        else:
            actions = self.GetAllPossibleActions_SpecialTurns(gameState, player)
        
        if type(actions) != list:
            actions = [actions]
        return actions

    # Return selected action
    def DoMove(self, game):

        if game.gameState.currPlayer != self.seatNumber and game.gameState.currState != "WAITING_FOR_DISCARDS":
            #raise Exception("\n\nReturning None Action - INVESTIGATE\n\n")
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        if len(possibleActions) == 1:
            return possibleActions[0]
        
        randIndex = random.randint(0, len(possibleActions)-1)
        chosenAction = possibleActions[randIndex]
        
        if chosenAction.type == "MakeTradeOffer":
            self.tradeCount += 1
        
        return chosenAction
        # NOTE: If no actions returned should we return EndTurn/RollDice?

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

    # TODO: now only offering 2 of a single resource for 1, should be able to offer 2 different for 1
    def GetPossiblePlayerTrades(self, gameState, player):

        if player is None:
            player = self

        possibleTrades = []

        if sum(player.resources) > 0:
            for giveIndex in range(5):
                for giveAmount in [1, 2]:
                    if player.resources[giveIndex] >= giveAmount:
                        giveResources    = [0, 0, 0, 0, 0, 0]
                        giveResources[giveIndex] = giveAmount
                        for getIndex in range(5):
                            if getIndex != giveIndex:
                                getResources    = [0, 0, 0, 0, 0, 0]
                                getResources[getIndex] = 1
                                # Go through other players and only offer to players who have resource
                                toPlayers = [True, True, True, True]
                                toPlayers[player.seatNumber] = False
                                for otherPlayer in gameState.players:
                                    if otherPlayer.seatNumber != player.seatNumber:
                                        if otherPlayer.resources[getIndex] == 0:
                                            toPlayers[otherPlayer.seatNumber] = False
                                            # print(f"{self.seatNumber} Cannot offer trade: {giveResources}_{getResources} to {player.seatNumber}, since resources: {player.resources[:5]}")
                                if any(toPlayers):
                                    # print(f"{player.seatNumber}(Turn {player.stats.numTurns}) with resources: {player.resources[:5]}, Offer: {giveResources[:5]}_{getResources[:5]}")
                                    tradeAction = MakeTradeOfferAction(fromPlayerNumber=self.seatNumber,
                                                                    toPlayers=toPlayers,
                                                                    giveResources=giveResources, getResources=getResources)
                                    possibleTrades.append(tradeAction)
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

    def GetPossiblePlayerTradeReactions(self, gameState, player):

        canTrade = True
        rejectTrade = RejectTradeOfferAction(playerNumber=player.seatNumber)

        for i in range(5):
            if player.resources[i] < gameState.currTradeOffer.getResources[i]:
                return [rejectTrade]
        else:
            acceptTrade = AcceptTradeOfferAction(playerNumber=player.seatNumber, offerPlayerNumber=gameState.currTradeOffer.fromPlayerNumber, gameState=gameState)
            return [acceptTrade, rejectTrade]