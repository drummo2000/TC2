import subprocess
import os
import signal
import time
import argparse

from Agents.AgentRandom import *
from Agents.AgentMCTS import AgentMCTS
from Agents.AgentUCT import AgentUCT
from Agents.AgentUCTTuned import AgentUCTTuned
from Agents.AgentRandom2 import AgentRandom2
from Agents.AgentModel import AgentModel, AgentMultiModel
from DeepLearning.PPO import MaskablePPO
from DeepLearning.GetObservation import getObservationSimplified
from DeepLearning.GetActionMask import getActionMaskTrading

import CatanData.CSVGenerator
from Client import Client
from tabulate import tabulate
from DeepLearning.Stats import headers
import pandas as pd

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

class TC2Main(object):

    def __init__(self, serverType = ""):

        self.serverProcess = None
        self.robot1Process = None
        self.robot2Process = None
        self.robot3Process = None
        self.clientProcess = None
        self.player        = None
        self.ourClient:Client     = None
        self.simCount      = 1000
        self.serverType    = serverType

    def ComposeGameStatsMessage(self, gameState):

        msg =  "#########################################################\n" \
              +"Game Over! Player {0} Wins!\n".format(gameState.players[gameState.winner].name) \
              +"GAME STATS:\n" \
              +" total turns: {0} \n starting player: {1} \n largest army player: {2} \n longest road player: {3}\n".format(
                    gameState.currTurn,
                    gameState.startingPlayer,
                    gameState.largestArmyPlayer,
                    gameState.longestRoadPlayer
                )\
              +"#########################################################\n"

        for n in range(0, len(gameState.players)):
            msg += "Player {0} stats:".format(gameState.players[n].name)\
                +  "Player {0} is a {1} agent".format(gameState.players[n].name,
                                                      gameState.players[n].agentName) \
                +    "his resources are: "       \
                     "\n POINTS          = {0} " \
                     "\n LARGEST ARMY    = {1} " \
                     "\n LONGEST ROAD    = {2} " \
                     "\n RESOURCES       = {3} " \
                     "\n PIECES          = {4} " \
                     "\n KNIGHTS         = {5} " \
                     "\n DICE PRODUCTION = {6}".format(
                        gameState.players[n].GetVictoryPoints(),
                        gameState.players[n].biggestArmy,
                        gameState.players[n].biggestRoad,
                        gameState.players[n].resources,
                        gameState.players[n].numberOfPieces,
                        gameState.players[n].knights,
                        gameState.players[n].diceProduction
                    )\

            devCards = ""

            for j in range(0, len(g_developmentCards)):
                devCards += " {0} : {1}".format(
                    g_developmentCards[j], gameState.players[n].developmentCards[n]
                )

            msg += "DevCards : " + devCards

            msg += " Roads: {0}\n Settlements: {1}\n Cities: {2}".format(
                    [hex(road) for road in gameState.players[n].roads],
                    [hex(settlement) for settlement in gameState.players[n].settlements],
                    [hex(city) for city in gameState.players[n].cities])

            msg += "---------------------------------------------------------"

            return msg

    def SaveGameStats(self, gameState):

        msg = self.ComposeGameStatsMessage(gameState)

        if os.path.isfile("SimulatorLogs/JSettlersVSGames.txt"):

            with open("SimulatorLogs/JSettlersVSGames.txt", "a") as text_file:
                text_file.write("\n"+msg)
        else:

            with open("SimulatorLogs/JSettlersVSGames.txt", "w") as text_file:
                text_file.write(msg)


    def RunClient(self, killProcess=True, host="8880"):
        result = self.ourClient.StartClient(("localhost", int(host)))
        return result


    def InitGame(self, host = "8880", canInitServer = True, gameNamePrefix = None, callProcess=True):

        model=MaskablePPO.load("DeepLearning/Thesis/5.Opponents/Models/Distribution/model_14966784.zip")
        self.player = AgentModel("TC2_agent", 0, playerTrading=False, recordStats=True, jsettlersGame=True, model=model)


        # Change the current directory...
        mycwd = os.getcwd()

        os.chdir("..")
        os.chdir('JSettlers-1.0.6')

        # Double negation in the switches here are a bit confusing TBH...
        serverType = '_perfectInfo'
        if True and canInitServer:

            self.serverProcess = subprocess.Popen(f"java -jar JSettlersServer{serverType}.jar {host} 10 dbUser dbPass",
                                             shell=True, stdout=subprocess.PIPE)

            time.sleep(3)

        if True and callProcess:
            self.robot1Process = subprocess.Popen(
                f"java -cp JSettlersServer{self.serverType}.jar soc.robot.SOCRobotClient localhost {host} robot1 passwd",
                shell=True, stdout=subprocess.PIPE)

            self.robot2Process = subprocess.Popen(
                f"java -cp JSettlersServer{self.serverType}.jar soc.robot.SOCRobotClient localhost {host} robot2 passwd",
                shell=True, stdout=subprocess.PIPE)

            self.robot3Process = subprocess.Popen(
                f"java -cp JSettlersServer{self.serverType}.jar soc.robot.SOCRobotClient localhost {host} robot3 passwd",
                shell=True, stdout=subprocess.PIPE)

        if True and callProcess:

            self.clientProcess = subprocess.Popen(f"java -jar JSettlers.jar localhost {host}",
                shell=True, stdout=subprocess.PIPE)

        # Go back to the Client directory...
        os.chdir(mycwd)

        # if args.logging == 'i':
        #     logging.getLogger().setLevel(logging.INFO)
        # elif args.logging == 'd':
        #     logging.getLogger().setLevel(logging.DEBUG)

        #logging.getLogger().setLevel(logging.CRITICAL)

        gameName = "TestGame"
        if gameNamePrefix is not None:
            gameName += str(gameNamePrefix)

        # if not args.game:
        #     self.ourClient = Client(gameName, self.player, False, True)
        # else:
        self.ourClient = Client(gameName, self.player, True, True)

    if __name__ == '__main__':
        for i in range(10):
            from TC2Main import TC2Main

            main = TC2Main()

            host = str(int(sys.argv[1])+i)

            print(host)

            main.InitGame(host=host)
            # Give some time so the server can start and the robots get in....
            time.sleep(3)

            main.RunClient(host=host)

            # os.system(f"lsof -ti:{host} | xargs kill -9")