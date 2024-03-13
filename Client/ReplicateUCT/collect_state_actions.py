from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentMCTS import AgentMCTS
from Agents.AgentUCT import AgentUCT
from Agents.AgentModel import AgentMultiModel, AgentModel
from Game.CatanGame import *
from CatanSimulator import CreateGame
from DeepLearning.PPO import MaskablePPO
from Game.CatanPlayer import PlayerStatsTracker
from tabulate import tabulate
from DeepLearning.Stats import headers
import dill as pickle
import pandas as pd
from CatanData.GameStateViewer import SaveGameStateImage, DisplayImage
import time
import math
from DeepLearning.GetObservation import getObservationSimplified, getObservationFull
import joblib
from DeepLearning.GetActionMask import allActionsDict
import copy


dummyAgent = AgentRandom2("Dummy", -1)

winner = [0,0,0,0]

players = [ AgentUCT("P0", 0, simulationCount=100),
            AgentRandom2("P1", 1),
            AgentRandom2("P2", 2),
            AgentRandom2("P3", 3)]

for episode in range(1000):
    game = CreateGame(players)
    game = pickle.loads(pickle.dumps(game, -1))
    actionNum = 0

    while True:
        currPlayer = game.gameState.players[game.gameState.currPlayer]
        agentAction = currPlayer.DoMove(game)

        if currPlayer.seatNumber == 0:
            # if only one possible action skip recording
            actions = dummyAgent.GetPossibleActions(gameState=game.gameState, player=currPlayer)
            if len(actions) != 1:
                actionIndex = allActionsDict[agentAction.getString()]
                joblib.dump((game.gameState, actionIndex), f"StateActions2/StateAction_{episode}_{actionNum}.joblib")
                actionNum += 1
        
        # Apply the action Changing the gameState
        agentAction.ApplyAction(game.gameState)

        if game.gameState.currState == "OVER":
            break
       
    winner[game.gameState.winner] += 1

print("\n\nWinnings: ", winner)

# state, action = joblib.load("StateActions/StateAction_1_0.joblib")
# print(state.currState)
# print(*getObservationFull(state))