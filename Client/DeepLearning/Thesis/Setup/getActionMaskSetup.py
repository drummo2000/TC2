from Game.CatanGame import constructableNodesList, constructableHexesList, constructableEdgesList
from Game.CatanAction import *
from itertools import combinations
import numpy as np

buildRoadActions = []
for edgeNumber in constructableEdgesList:
    buildRoadActions.append(f"BuildRoad{edgeNumber}")

buildSettlementActions = []
for nodeNumber in constructableNodesList:
    buildSettlementActions.append(f"BuildSettlement{nodeNumber}")

setupActionList = [*buildRoadActions, *buildSettlementActions]


setupActionsDict = {action: index for index, action in enumerate(setupActionList)}


# For now only do settlements
def getSetupActionMask(possibleActions: list[Action]):
    indexActionDict = {}

    mask = [0] * len(setupActionList)
    for action in possibleActions:
        mask[setupActionsDict[action.getString()]] = 1
        indexActionDict[setupActionsDict[action.getString()]] = action
    return np.array(mask), indexActionDict

