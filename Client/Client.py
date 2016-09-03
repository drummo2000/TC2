import socket
import logging
from JSettlersMessages import *
from CatanPlayer import *
from CatanGame import *

class Client:

    def __init__(self, gameName, player, autoStart):

        self.socket       = None
        self.game         = None

        self.joinedAGame  = False
        self.isSeated     = False
        self.gameStarted  = False

        self.gameName     = gameName
        self.player       = player

        self.autoStart    = autoStart
        self.botsInit     = False

        self.messagetbl = {}
        for g in globals():
            cg = globals()[g]
            if g.endswith("Message") and hasattr(cg, "id"):
                self.messagetbl[str(cg.id)] = (cg, g)

    # Connection to jsettlers game server
    def ConnectToServer(self, serverAddress):

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            self.socket.connect(serverAddress)

            self.socket.settimeout(240)

        except socket.error, exc:

            logging.critical("Caught exception socket.error : %s" % exc)

            logging.critical("Could Not Connect to JSettlers Server :(")

            return False

        logging.info("Connected to JSettlers!")

        return True

    def StartClient(self, serverAddress):

        if self.ConnectToServer(serverAddress):
            while True:
                result = self.Update()
                if result is not None:
                    return result

    def CreateMessage(self, raw_msg):

        highByte = chr(len(raw_msg) / 256)
        lowByte = chr(len(raw_msg) % 256)

        return highByte + lowByte + raw_msg

    def ParseMessage(self, message):
        """ Create a message from recieved data """
        id, txt = message[:4], message[5:]

        if not id in self.messagetbl:
            logging.critical("Can not parse '{0}'".format(message))
            return

        messageClass, messageName = self.messagetbl[id]
        inst = messageClass.parse(txt)

        return (messageName, inst)

    def SendMessage(self, message):

        logging.debug("Sending: {0}".format(message.to_cmd()))

        self.socket.send(self.CreateMessage(message.to_cmd()))

    def Update(self):

        def recvwait(size):
            sofar = 0
            r = ""
            while True:
                r += self.socket.recv(size - len(r))
                if len(r) >= size:
                    break
            return r

        try:
            highByte = ord(recvwait(1))
            lowByte = ord(recvwait(1))
            transLength = highByte * 256 + lowByte
            msg = recvwait(transLength)

            logging.debug("Received this from JSettlers: {0}".format(msg))

        except socket.timeout:
            logging.critical("recv operation timed out.")
            return -1

        try:
            parsed = self.ParseMessage(msg)
        except:
            logging.critical("Failed to parse this message: {0}".format(msg))
            self.socket.close()
            return -1

        if parsed is None:
            logging.debug("Message not supported -- {0}".format(msg))
            return None
        else:
            (messageName, message) = parsed
            self.TreatMessage(messageName, message)

    def TreatMessage(self, name, instance):

        if   name == "ChannelsMessage":

            logging.info("There are {0} channels available: {1}".format(len(instance.channels), instance.channels))

        elif name == "GamesMessage":

            logging.info("There are {0} games available: {1}".format(len(instance.games), instance.games))

            if not self.joinedAGame:
                logging.info("Starting a new game...")
                message = JoinGameMessage(self.player.name, "", socket.gethostname(), self.gameName)
                self.SendMessage(message)

        elif name == "NewGameMessage":

            logging.info("Crated game: '{0}'".format(instance.gameName))

        elif name == "JoinGameAuthMessage":

            logging.info("Entered game: '{0}'".format(instance.gameName))

            self.joinedAGame = True

            self.game = Game(GameState())

            if not self.isSeated:
                logging.info("Sitting on seat number {0}".format(self.player.seatNumber))
                message = SitDownMessage(self.gameName, self.player.name, self.player.seatNumber, True)
                self.SendMessage(message)

        elif name == "SitDownMessage":

            self.game.AddPlayer(Player(instance.nickname, instance.playerNumber))

        elif name == "ChangeFaceMessage":

            self.isSeated = True

            if not self.gameStarted:
                logging.info("Seated. Starting game...")

                self.gameStarted = True

                message1 = ChangeFaceMessage(self.gameName, self.player.seatNumber, 44)
                self.SendMessage(message1)

                if self.autoStart:
                    message2 = StartGameMessage(self.gameName)
                    self.SendMessage(message2)

        elif name == "GameMembersMessage":

            logging.info("Players in this game: {0}".format(instance.members))

        elif name == "BoardLayoutMessage":

            logging.info("Received board layout")

            logging.debug("Board Hexes   = {0}".format(instance.hexes))
            logging.debug("Board Numbers = {0}".format(instance.numbers))

            self.game.CreateBoard(instance)

        elif name == "LongestRoadMessage":

            logging.info("Received longest road player: {0}".format(instance.playerNumber))

            self.game.gameState.longestRoadPlayer = int(instance.playerNumber)

        elif name == "LargestArmyMessage":

            logging.info("Received largest army player: {0}".format(instance.playerNumber))

            self.game.gameState.largestArmyPlayer = int(instance.playerNumber)

        elif name == "PlayerElementMessage":

            logging.info("Player seated on {0} : {1} {2}, amount: {3}".format(instance.playerNumber, instance.action, instance.element, instance.value))

        elif name == "GameStateMessage":

            logging.info("Switching gameState from {0} to: {1}".format(self.game.gameState.currState, instance.stateName))

            self.game.gameState.currState = instance.stateName

            if instance.stateName == "START1A":

                logging.info("Current Players Are: {0}".format([player.name for player in self.game.gameState.players]))

            if instance.stateName == "OVER":
                pass


logging.getLogger().setLevel(logging.INFO)
#logging.getLogger().setLevel(logging.DEBUG) # FOR DEBUG

client = Client("TestGame", Player("Danda", 0), True)
client.StartClient(("localhost", 8880))