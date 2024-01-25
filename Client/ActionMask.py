from CatanGame import constructableNodesList, constructableHexesList, constructableEdgesList
from CatanAction import *
from itertools import combinations
import numpy as np

buildRoadActions = []
for edgeNumber in constructableEdgesList:
    buildRoadActions.append(f"BuildRoad{edgeNumber}")

buildSettlementActions = []
for nodeNumber in constructableNodesList:
    buildSettlementActions.append(f"BuildSettlement{nodeNumber}")

buildCityActions = []
for nodeNumber in constructableNodesList:
    buildCityActions.append(f"BuildCity{nodeNumber}")

rollDiceAction = "RollDices"
buyDevelopmentCardAction = "BuyDevelopmentCard"
useKnightsCardAction = "UseKnightsCard"

monopolyCardActions = []
for i in range(5):
    monopolyCardActions.append(f"UseMonopolyCard{i}")

yearOfPlentyCardActions = []
for i in range(0, 5):
    for j in range(i, 5):
        chosenResources = [0, 0, 0, 0, 0]
        chosenResources[i] += 1
        chosenResources[j] += 1
        yearOfPlentyCardActions.append(f"UseYearOfPlentyCard{chosenResources[0]}{chosenResources[1]}{chosenResources[2]}{chosenResources[3]}{chosenResources[4]}")

useFreeRoadsCardAction = "UseFreeRoadsCard"

placeRobberActions = []
for hexNumber in constructableHexesList:
    placeRobberActions.append(f"PlaceRobber{hexNumber}")

choosePlayerToStealFromActions = []
for i in range(1, 4):
    choosePlayerToStealFromActions.append(f"ChoosePlayerToStealFrom{i}")

endTurnAction = "EndTurn"

discardResourcesActions = ["DiscardResources000000"]
# All possible combinations of picking 4 and 5 resources
resourcesPopulation =   [0 for i in range(0, 5)] + \
                        [1 for j in range(0, 5)] + \
                        [2 for k in range(0, 5)] + \
                        [3 for l in range(0, 5)] + \
                        [4 for m in range(0, 5)]

possibleSelections = set(combinations(resourcesPopulation, 4)).union(combinations(resourcesPopulation, 5))
for selection in possibleSelections:
    discardResourcesActions.append(f"DiscardResources{selection.count(0)}{selection.count(1)}{selection.count(2)}{selection.count(3)}{selection.count(4)}0")
discardResourcesActions.append("DiscardResources>5")

bankTradeOfferActions = []
# for each give resource 5, theres 3 possible trade rates and 4 possible get resources

tradeRates = [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]]
for tradeRate in tradeRates:
    for i in range(5):
        give = [0, 0, 0, 0, 0]
        give[i] = tradeRate[i]
        for j in range(1, 5):
            get = [0, 0, 0, 0, 0]
            index = (i + j) % 5
            get[index] = 1
            bankTradeOfferActions.append(f"BankTradeOffer{give[0]}{give[1]}{give[2]}{give[3]}{give[4]}_{get[0]}{get[1]}{get[2]}{get[3]}{get[4]}")

allActionsList = [*buildRoadActions,
                  *buildSettlementActions,
                  *buildCityActions,
                  rollDiceAction,
                  buyDevelopmentCardAction,
                  useKnightsCardAction,
                  *monopolyCardActions,
                  *yearOfPlentyCardActions,
                  useFreeRoadsCardAction,
                  *placeRobberActions,
                  *choosePlayerToStealFromActions,
                  endTurnAction,
                  *discardResourcesActions,
                  *bankTradeOfferActions
                  ]

# print(len(allActionsList))

# print(f"BuildRoad: {len(buildRoadActions)}")
# print(f"buildSettlementActions: {len(buildSettlementActions)}")
# print(f"buildCityActions: {len(buildCityActions)}")
# print(f"monopolyCardActions: {len(monopolyCardActions)}")
# print(f"yearOfPlentyCardActions: {len(yearOfPlentyCardActions)}")
# print(f"placeRobberActions: {len(placeRobberActions)}")
# print(f"choosePlayerToStealFromActions: {len(choosePlayerToStealFromActions)}")
# print(f"discardResourcesActions: {len(discardResourcesActions)}")
# print(f"bankTradeOfferActions: {len(bankTradeOfferActions)}")

# e.g: {BuildRoad77: 5, BuildRoad79: 6, ...}
allActionsDict = {action: index for index, action in enumerate(allActionsList)}

# Takes in list of possible actions and returns action mask for Network
def getActionMask(possibleActions: list[Action]):
    # create new dictionary: {57: Action(), 68: Action()}
    indexActionDict = {}

    mask = [0] * 485
    for action in possibleActions:
        mask[allActionsDict[action.getString()]] = 1
        indexActionDict[allActionsDict[action.getString()]] = action
    return np.array(mask), indexActionDict
