from AgentRandom import *
from datetime import datetime
from datetime import timedelta
import copy
import cPickle

class AgentMCTS(AgentRandom):

    DiceProbability = \
    {
        2  : listm([0.03, 0.03, 0.03, 0.03, 0.03, 0.03]),
        3  : listm([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]),
        4  : listm([0.08, 0.08, 0.08, 0.08, 0.08, 0.08]),
        5  : listm([0.11, 0.11, 0.11, 0.11, 0.11, 0.11]),
        6  : listm([0.14, 0.14, 0.14, 0.14, 0.14, 0.14]),
        7  : listm([0.17, 0.17, 0.17, 0.17, 0.17, 0.17]),
        8  : listm([0.14, 0.14, 0.14, 0.14, 0.14, 0.14]),
        9  : listm([0.11, 0.11, 0.11, 0.11, 0.11, 0.11]),
        10 : listm([0.08, 0.08, 0.08, 0.08, 0.08, 0.08]),
        11 : listm([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]),
        12 : listm([0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
    }

    # * OPTION (for performance)*
    #    implement nodes as a tuple (worse to read/understand)
    # MCTS TREE NODE STRUCTURE:
    # (gameState, action     , Q-value , N-value     , PARENT     , CHILDREN)
    #  currState, from parent, reward  , n. of visits, parent node, children
    class MCTSNode:

        def __init__(self, player, state, action, qValue, nValue, parent, children, actionsFunction):

            self.actingPlayer    = player
            self.gameState       = cPickle.dumps(state, -1) # current gameState
            self.action          = action  # action that led to this state
            self.QValue          = qValue  # node estimated reward value
            self.NValue          = nValue  # number of visits
            self.parent          = parent  # parent
            self.children        = children  # children
            self.possibleActions = actionsFunction(state,
                                                   state.players[state.currPlayer])
            self.currentPlayer   = state.currPlayer
            self.isTerminal      = state.IsTerminal()

        def GetState(self):
            if self.NValue < AgentMCTS.saveNodeValue:
                return cPickle.loads(self.gameState)
            else:
                return self.gameState

        def GetStateCopy(self):
            if self.NValue < AgentMCTS.saveNodeValue:
                return cPickle.loads(self.gameState)
            else:
                return cPickle.loads(cPickle.dumps(self.gameState, -1))

        def UpdateNValue(self):

            self.NValue += 1
            if self.NValue == AgentMCTS.saveNodeValue:
                self.gameState = cPickle.loads(self.gameState)

    explorationConstant = 1.0

    saveNodeValue       = 20

    def __init__(self, name, seatNumber, choiceTime = 10.0, simulationCount = None):

        super(AgentMCTS, self).__init__(name, seatNumber)

        self.choiceTime          = choiceTime
        self.agentName           = "MONTE CARLO TREE SEARCH : {0} sec".format(choiceTime)
        self.simulationCounter   = 0
        self.maxSimulations      = simulationCount
        self.movesToDo           = []

    def DoMove(self, game):

        # SERVER SPECIAL CASES:
        # If its not our turn and the server is not waiting for discards...
        if game.gameState.currPlayer != self.seatNumber and \
            game.gameState.currState != "WAITING_FOR_DISCARDS":
            return None
        # If the server is waiting for discards, respond, if needed...
        if game.gameState.currState == "WAITING_FOR_DISCARDS":
            return self.ChooseCardsToDiscard()
        # If we already done our setup phase, ignore repeated message (strange bug from server)...
        if (game.gameState.currState == "START1A" and self.firstSettlementBuild) or \
           (game.gameState.currState == "START1B" and self.firstRoadBuild) or \
           (game.gameState.currState == "START2A" and self.secondSettlementBuild) or \
           (game.gameState.currState == "START2B" and self.secondRoadBuild):
            return None

        # SPECIAL CASE -> WE JUST GOT OUR TURN AND CAN CLEAR THE "BUFFER"...
        if game.gameState.currState == "PLAY":

            print("empty buffer! -> PLAY")

            self.movesToDo = []

        # IF I HAVE MOVES IN MY "BUFFER", RETURN THOSE...
        if len(self.movesToDo) > 0:

            print("Accessed Move Buffer = {0}".format(self.movesToDo))

            action = self.movesToDo[0]  # get first element
            # SPECIAL CASE -> MONOPOLY ACTION - we don't know what resources will come from the server
            if isinstance(action, UseDevelopmentCardAction) and \
                action.index == g_developmentCards.index('MONOPOLY'):
                print("Clear buffer -> MONOPOLY")
                self.movesToDo = []
            else:
                self.movesToDo = self.movesToDo[1:] # remove first element

            print("BUFFER ACTION = \n{0}".format(action))

            return action

        self.simulationCounter = 0

        state = cPickle.loads(cPickle.dumps(game.gameState, -1))

        AgentMCTS.PrepareGameStateForSimulation(state)

        action = self.MonteCarloTreeSearch(state, timedelta(seconds=self.choiceTime), self.maxSimulations)

        # SPECIAL CASE -> MONOPOLY ACTION - we don't know what resources will come from the server
        if isinstance(action, UseDevelopmentCardAction) and \
                        action.index == g_developmentCards.index('MONOPOLY'):

            print("empty buffer! -> MONOPOLY")

            self.movesToDo = []

        return action

    def MonteCarloTreeSearch(self, gameState, maxDuration, simulationCount):

        rootNode = self.MCTSNode(
                        player=self.seatNumber,
                        state=gameState,
                        action=None,
                        qValue=listm(0 for i in range(len(gameState.players))),
                        nValue=0,
                        parent=None,
                        children=[],
                        actionsFunction=self.GetPossibleActions)

        print("GAME STATE      : {0}".format(gameState.currState))
        print("POSSIBLE ACTIONS: {0}".format(rootNode.possibleActions))

        print("RESOURCES = BRICK: {0}\n"
              "              ORE: {1}\n"
              "             WOOL: {2}\n"
              "            GRAIN: {3}\n"
              "           LUMBER: {4}".format(
            self.resources[0],
            self.resources[1],
            self.resources[2],
            self.resources[3],
            self.resources[4]
        ))

        if sum(self.developmentCards) > 0:
            print("DEVELOPMENT CARDS = KNIGHT:        {0}\n"
                  "                    ROAD_BUILDING: {1}\n"
                  "                    YEAR_OF_PLENTY:{2}\n"
                  "                    MONOPOLY:      {3}\n"
                  "                    VICTORY_POINT: {4}".format(
                self.developmentCards[0],
                self.developmentCards[1],
                self.developmentCards[2],
                self.developmentCards[3],
                self.developmentCards[4]
            ))

        if rootNode.possibleActions is None:
            print("MCTS ERROR! POSSIBLE ACTIONS FROM ROOT NODE ARE NONE!!!!")
            return None

        elif len(rootNode.possibleActions) == 1:
            return rootNode.possibleActions[0]

        elif len(rootNode.possibleActions) <= 0:
            print("MCTS ERROR! NO POSSIBLE ACTIONS FROM ROOT NODE!")
            return None

        startTime = datetime.utcnow()

        # if a simulation count is given, ignore time constraints
        def Condition():
            if simulationCount is None:
                return (datetime.utcnow() - startTime) < maxDuration
            else:
                return self.simulationCounter < simulationCount

        while Condition():

            nextNode = self.TreePolicy(rootNode)
            reward   = self.SimulationPolicy(nextNode.GetStateCopy())
            self.BackUp(nextNode, reward)
            self.simulationCounter += 1

        #print("TOTAL SIMULATIONS = {0}".format(self.simulationCounter))
        #print("TOTAL TIME        = {0}".format((datetime.utcnow() - startTime)))

        best = self.BestChild(rootNode, 0)

        # KEEP FUTURE ACTIONS IN A "BUFFER"...
        # IF WE ARE CHOOSING A ROBBER POSITION, DON'T KEEP BUFFER -> WE DON'T KNOW WHAT RESOURCE WE WILL STEAL!
        if gameState.currState != 'PLACING_ROBBER':

            bestChild = self.BestChild(best, 0)
            while bestChild is not None and \
                  bestChild.actingPlayer == self.seatNumber and \
                  not bestChild.isTerminal:

                self.movesToDo.append(bestChild.action)

                bestChild = self.BestChild(bestChild, 0)

            print("Created Move Buffer = {0}".format(self.movesToDo))

        print("CHOSEN ACTION = \n{0}".format(best.action))

        return best.action

    def TreePolicy(self, node):

        while node.isTerminal is False and node.possibleActions is not None:
            # There are still actions to try in this node...
            if len(node.possibleActions) > 0:
                return self.Expand(node)

            node = self.BestChild(node, AgentMCTS.explorationConstant)

        return node

    def Expand(self, node):

        chosenAction = random.choice(node.possibleActions)

        node.possibleActions.remove(chosenAction)

        nextGameState = node.GetStateCopy()

        chosenAction.ApplyAction(nextGameState)

        estimatedQValues = AgentMCTS.GetEstimatedQValues(len(nextGameState.players), chosenAction, nextGameState.players[node.currentPlayer])

        childNode = self.MCTSNode(player=node.currentPlayer,
                                  state=nextGameState,
                                  action=chosenAction,
                                  qValue=estimatedQValues,
                                  nValue=0,
                                  parent=node,
                                  children=[],
                                  actionsFunction=self.GetPossibleActions)

        node.children.append(childNode)

        return childNode

    def BestChild(self, node, explorationValue, player=None):

        if len(node.children) <= 0:
            return None

        # Returns the Child Node with the max 'Q-Value'
        #return max(node.children, key=lambda child: child.QValue[currPlayerNumber])

        # Returns the Child according to the UCB Function
        def UCB1(childNode):

            tgtPlayer = node.currentPlayer if player is None else player

            evaluationPart  = float(childNode.QValue[tgtPlayer]) / float(childNode.NValue)
            explorationPart = explorationValue * math.sqrt( (2 * math.log(node.NValue)) / float(childNode.NValue) )
            return evaluationPart + explorationPart

        return max(node.children, key=lambda child : UCB1(child))

    def SimulationPolicy(self, gameState):

        while not gameState.IsTerminal():

            possibleActions = self.GetPossibleActions(gameState,
                                                      gameState.players[gameState.currPlayer],
                                                      atRandom=True)
            if len(possibleActions) > 1:
                action = random.choice(possibleActions)
            else:
                action = possibleActions[0]

            action.ApplyAction(gameState)

        return self.Utility(gameState)

    def BackUp(self, node, reward):

        while node is not None:
            node.UpdateNValue()
            node.QValue += reward
            node         = node.parent

    def Utility(self, gameState):

        vp = listm(0 for i in range(len(gameState.players)))

        # 60 % WINS!!!!
        for player in gameState.players:

            resourcesVal = AgentMCTS.GetResourceUsabilityValue(player)
            vp[player.seatNumber] += (player.GetVictoryPoints(forceUpdate=True) / 10.0) # + (0.5 * resourcesVal)

        vp[gameState.winner] += 1

        # for player in gameState.players:
        #     vp[player.seatNumber] = self.GetGameStateReward(gameState, player)

        return vp

    @staticmethod
    def GetResourceUsabilityValue(player):

        totalProbabilities = [0, 0, 0, 0, 0, 0]
        for number, production in player.diceProduction.iteritems():
            totalProbabilities += production * AgentMCTS.DiceProbability[number]

        cityProb       = sum(BuildCityAction.cost * totalProbabilities)
        settlementProb = sum(BuildSettlementAction.cost * totalProbabilities)
        roadProb       = sum(BuildRoadAction.cost * totalProbabilities)
        cardProb       = sum(BuyDevelopmentCardAction.cost * totalProbabilities)

        return cityProb * 0.3 + settlementProb * 0.3 + roadProb * 0.2 + cardProb * 0.2

    @staticmethod
    def GetActionEstimatedValue(action, player):

        if action.type == 'BuildRoad':
            if player.possibleSettlements <= 0:
                return 20
            return 10/(len(player.roads) + 1)
        if action.type == 'BuildSettlement':
            return 50
        if action.type == 'BuildCity':
            return 100
        if action.type == 'BuyDevelopmentCard':
            if player.biggestArmy:
                return 1
            return 5
        if action.type == 'BankTradeOffer':
            if not player.CanAfford(BuildCityAction.cost) and \
               not player.CanAfford(BuildSettlementAction.cost) and \
               not player.CanAfford(BuildRoadAction.cost) and \
               not player.CanAfford(BuyDevelopmentCardAction.cost):
                return 10
            return 1

        return 0

    @staticmethod
    def GetEstimatedQValues(lenght, action, player):

        estimatedQValues = listm(0 for i in range(lenght))

        estimatedQValues[player.seatNumber] = AgentMCTS.GetActionEstimatedValue(action, player)

        return estimatedQValues

    # def GetGameStateReward(self, gameState, player):
    #
    #     playerPoints   = player.GetVictoryPoints()
    #
    #     playerPoints  *= 3 if gameState.winner == player.seatNumber else 1
    #
    #     longestRoadPts = 3 if gameState.longestRoadPlayer == player.seatNumber else 0
    #
    #     largestArmyPts = 3 if gameState.largestArmyPlayer == player.seatNumber else 0
    #
    #     numSettlements = len(player.settlements)
    #
    #     numCities      = len(player.cities)
    #
    #     return  playerPoints + (numSettlements * 2) + (numCities * 3) + \
    #             largestArmyPts + longestRoadPts

    @staticmethod
    def PrepareGameStateForSimulation(gameState):

        for player in gameState.players:

            if player is None:
                continue

            quantity = player.resources[g_resources.index('UNKNOWN')]

            if quantity > 0:

                player.resources[g_resources.index('UNKNOWN')] = 0

                resources = [0, 0, 0, 0, 0, 0]

                for i in range(0, quantity):
                    resources[random.randint(0, 4)] += 1

                player.resources = player.resources + resources

    def GetPossibleActions(self, gameState, player, atRandom=False):

        if not gameState.setupDone:
            return self.GetPossibleActions_SetupTurns(gameState, player)
        elif gameState.currState == "PLAY":
            return self.GetPossibleActions_PreDiceRoll(player)
        elif gameState.currState == "PLAY1":
            if atRandom:
                return [self.GetRandomAction_RegularTurns(gameState, player)]
            else:
                return self.GetPossibleActions_RegularTurns(gameState, player)
        else:
            return self.GetPossibleActions_SpecialTurns(gameState, player, atRandom)

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

    def ChooseRobberPosition(self, gameState, player):

        playerHexes = []
        for s in player.settlements:
            playerHexes += gameState.boardNodes[s].adjacentHexes
        for c in player.cities:
            playerHexes += gameState.boardNodes[c].adjacentHexes

        def CheckHex(hexPosition):

            if hexPosition in playerHexes or \
               hexPosition == gameState.robberPos:
                return False
            return True

        possiblePositions = []

        for otherPlayer in gameState.players:

            if otherPlayer.seatNumber == player.seatNumber:
                continue

            for settlementPos in otherPlayer.settlements:
                for hexPos in gameState.boardNodes[settlementPos].adjacentHexes:
                    if hexPos in gameState.possibleRobberPos and CheckHex(hexPos):
                        possiblePositions.append(hexPos)

            for cityPos in otherPlayer.cities:
                for hexPos in gameState.boardNodes[cityPos].adjacentHexes:
                    if hexPos in gameState.possibleRobberPos and CheckHex(hexPos):
                        possiblePositions.append(hexPos)

        return [PlaceRobberAction(player.seatNumber, pos) for pos in possiblePositions]

    def GetPossibleActions_SpecialTurns(self, gameState, player, atRandom):

        if gameState.currState == 'PLACING_ROBBER':

            if atRandom:
                return super(AgentMCTS, self).ChooseRobberPosition(gameState, player)
            else:
                return self.ChooseRobberPosition(gameState, player)

        elif gameState.currState == 'WAITING_FOR_DISCARDS':

            return [player.ChooseCardsToDiscard(player)]

        elif gameState.currState == 'WAITING_FOR_CHOICE':

            return [player.ChoosePlayerToStealFrom(gameState, player)]

        elif gameState.currState == "PLACING_FREE_ROAD1":

            possibleRoads = gameState.GetPossibleRoads(player)

            if possibleRoads is None or not possibleRoads or self.numberOfPieces[0] <= 0:
                return [ ChangeGameStateAction("PLAY1") ]

            return [BuildRoadAction(player.seatNumber, roadEdge,
                                    len(player.roads))
                    for roadEdge in possibleRoads]

        elif gameState.currState == "PLACING_FREE_ROAD2":

            possibleRoads = gameState.GetPossibleRoads(player)

            if possibleRoads is None or not possibleRoads or self.numberOfPieces[0] <= 0:
                return [ ChangeGameStateAction("PLAY1") ]

            return [BuildRoadAction(player.seatNumber, roadEdge,
                                    len(player.roads))
                    for roadEdge in possibleRoads]

    def GetPossibleActions_RegularTurns(self, gameState, player):

        if gameState.currState == 'PLAY':

            if not player.rolledTheDices and \
                    not player.playedDevCard and \
                    player.mayPlayDevCards[KNIGHT_CARD_INDEX] and \
                            player.developmentCards[KNIGHT_CARD_INDEX] > 0:
                return [UseKnightsCardAction(player.seatNumber, None, None)]

            if not player.rolledTheDices:
                return [RollDicesAction(player.seatNumber)]

        elif gameState.currState == 'PLAY1':

            possibleActions     = []
            possibleSettlements = gameState.GetPossibleSettlements(player)
            possibleRoads       = gameState.GetPossibleRoads(player)

            if player.settlements and \
                player.HavePiece(g_pieces.index('CITIES')) and \
                player.CanAfford(BuildCityAction.cost):

                possibleCities = gameState.GetPossibleCities(player)

                if possibleCities is not None and len(possibleCities) > 0:
                    possibleActions += [BuildCityAction(player.seatNumber, node, len(player.cities))
                                        for node in possibleCities]

            if player.HavePiece(g_pieces.index('SETTLEMENTS')) and \
                    player.CanAfford(BuildSettlementAction.cost) and \
                    possibleSettlements:

                possibleActions += [BuildSettlementAction(player.seatNumber, node, len(player.settlements))
                                    for node in possibleSettlements]

            if player.HavePiece(g_pieces.index('ROADS')) and \
                    player.CanAfford(BuildRoadAction.cost) and \
                    possibleRoads:

                possibleActions += [BuildRoadAction(player.seatNumber, edge, len(player.roads))
                                    for edge in possibleRoads]

            if gameState.CanBuyADevCard(player) and not player.biggestArmy:

                possibleActions += [BuyDevelopmentCardAction(player.seatNumber)]

            if not player.playedDevCard and sum(player.developmentCards[:-1]) > 0 and \
                    not self.biggestArmy:

                possibleCardsToUse = []

                if not player.playedDevCard:

                    if player.developmentCards[MONOPOLY_CARD_INDEX] > 0 and \
                            player.mayPlayDevCards[MONOPOLY_CARD_INDEX]:

                            monopolyPick = player.GetMonopolyResource(gameState, player)

                            if monopolyPick is not None:
                                possibleCardsToUse += monopolyPick

                    if player.developmentCards[YEAR_OF_PLENTY_CARD_INDEX] > 0 and \
                            player.mayPlayDevCards[YEAR_OF_PLENTY_CARD_INDEX]:

                            yearOfPlentyPick = player.GetYearOfPlentyResource(gameState, player)

                            if yearOfPlentyPick is not None:
                                possibleCardsToUse += yearOfPlentyPick

                    if player.developmentCards[ROAD_BUILDING_CARD_INDEX] > 0 and \
                            player.mayPlayDevCards[ROAD_BUILDING_CARD_INDEX] and \
                                    player.numberOfPieces[0] > 0:

                            freeRoads = UseFreeRoadsCardAction(player.seatNumber, None, None)

                            if freeRoads is not None:
                                possibleCardsToUse.append(freeRoads)

                if possibleCardsToUse:
                    possibleActions += possibleCardsToUse

            possibleTrade = self.GetPossibleBankTrades(gameState, player)
            if possibleTrade is not None and possibleTrade:
                possibleActions += possibleTrade

            possibleActions += [EndTurnAction(playerNumber=player.seatNumber)]

            return possibleActions