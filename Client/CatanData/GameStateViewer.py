from Game.CatanGame import g_developmentCards

import pickle
from PIL import Image, ImageChops, ImageDraw, ImageFont
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
import time
from Game.CatanBoard import VICTORY_POINT_CARD_INDEX

def tintImage(image, tintColor):

    return ImageChops.multiply(image,
            Image.new('RGBA', image.size, tintColor))

citySettlementPos = {
    0x38 : (160, 53 ),
    0x5a : (225, 53 ),
    0x7c : (290, 53 ),
    0x27 : (128, 58 ),
    0x49 : (193, 58 ),
    0x6b : (258, 58 ),
    0x8d : (323, 58 ),
    0x36 : (128, 109),
    0x58 : (193, 109),
    0x7a : (258, 109),
    0x9c : (323, 109),
    0x25 : (95 , 124),
    0x47 : (160, 124),
    0x69 : (225, 124),
    0x8b : (290, 124),
    0xad : (355, 124),
    0x34 : (95 , 165),
    0x56 : (160, 165),
    0x78 : (225, 165),
    0x9a : (290, 165),
    0xbc : (355, 165),
    0x23 : (62 , 180),
    0x45 : (128, 180),
    0x67 : (193, 180),
    0x89 : (258, 180),
    0xab : (323, 180),
    0xcd : (388, 180),
    0x32 : (62 , 221),
    0x54 : (128, 221),
    0x76 : (193, 221),
    0x98 : (258, 221),
    0xba : (323, 221),
    0xdc : (388, 221),
    0x43 : (95 , 236),
    0x65 : (160, 236),
    0x87 : (225, 236),
    0xa9 : (290, 236),
    0xcb : (355, 236),
    0x52 : (95 , 277),
    0x74 : (160, 277),
    0x96 : (225, 277),
    0xb8 : (290, 277),
    0xda : (355, 277),
    0x63 : (128, 292),
    0x85 : (193, 292),
    0xa7 : (258, 292),
    0xc9 : (323, 292),
    0x72 : (128, 333),
    0x94 : (193, 333),
    0xb6 : (258, 333),
    0xd8 : (323, 333),
    0x83 : (160, 348),
    0xa5 : (225, 348),
    0xc7 : (290, 348)
}

straightRoadsPos = {
    0x26 : (127,  90),
    0x48 : (192,  90),
    0x6a : (257,  90),
    0x8c : (322,  90),
    0x24 : ( 95, 147),
    0x46 : (160, 147),
    0x68 : (225, 147),
    0x8a : (290, 147),
    0xac : (355, 147),
    0x22 : ( 62, 203),
    0x44 : (127, 203),
    0x66 : (192, 203),
    0x88 : (257, 203),
    0xaa : (322, 203),
    0xcc : (387, 203),
    0x42 : ( 95, 260),
    0x64 : (160, 260),
    0x86 : (225, 260),
    0xa8 : (290, 260),
    0xca : (355, 260),
    0x62 : (127, 317),
    0x84 : (192, 317),
    0xa6 : (257, 317),
    0xc8 : (322, 317)
}

rightRoadsPos = {
    0x27 : (144,  62),
    0x49 : (209,  62),
    0x6b : (274,  62),
    0x25 : (110, 118),
    0x47 : (175, 118),
    0x69 : (240, 118),
    0x8b : (305, 118),
    0x23 : ( 79, 175),
    0x45 : (144, 175),
    0x67 : (209, 175),
    0x89 : (274, 175),
    0xab : (339, 175),
    0x43 : (110, 232),
    0x65 : (175, 232),
    0x87 : (240, 232),
    0xa9 : (305, 232),
    0xcb : (370, 232),
    0x63 : (144, 288),
    0x85 : (209, 288),
    0xa7 : (274, 288),
    0xc9 : (339, 288),
    0x83 : (175, 346),
    0xa5 : (240, 346),
    0xc7 : (305, 346)
}

leftRoadsPos = {
    0x38 : (177,  62),
    0x5a : (242,  62),
    0x7c : (307,  62),
    0x36 : (144, 118),
    0x58 : (209, 118),
    0x7a : (274, 118),
    0x9c : (339, 118),
    0x34 : (112, 175),
    0x56 : (177, 175),
    0x78 : (242, 175),
    0x9a : (307, 175),
    0xbc : (372, 175),
    0x32 : (79 , 232),
    0x54 : (144, 232),
    0x76 : (209, 232),
    0x98 : (274, 232),
    0xba : (339, 232),
    0x52 : (112, 288),
    0x74 : (177, 288),
    0x96 : (242, 288),
    0xb8 : (307, 288),
    0x72 : (144, 346),
    0x94 : (209, 346),
    0xb6 : (274, 346)
}

hexPositions = {
    0x17 : (127,  34),
    0x39 : (192,  34),
    0x5b : (257,  34),
    0x7d : (322,  34),
    0x15 : (95 ,  90),
    0x37 : (160,  90),
    0x59 : (225,  90),
    0x7b : (290,  90),
    0x9d : (355,  90),
    0x13 : (62 , 146),
    0x35 : (127, 146),
    0x57 : (192, 146),
    0x79 : (257, 146),
    0x9b : (322, 146),
    0xbd : (387, 146),
    0x11 : (29 , 203),
    0x33 : (94 , 203),
    0x55 : (159, 203),
    0x77 : (224, 203),
    0x99 : (289, 203),
    0xbb : (354, 203),
    0xdd : (419, 203),
    0x31 : (62 , 260),
    0x53 : (127, 260),
    0x75 : (192, 260),
    0x97 : (257, 260),
    0xb9 : (322, 260),
    0xdb : (387, 260),
    0x51 : (95 , 317),
    0x73 : (160, 317),
    0x95 : (225, 317),
    0xb7 : (290, 317),
    0xd9 : (355, 317),
    0x71 : (127, 373),
    0x93 : (192, 373),
    0xb5 : (257, 373),
    0xd7 : (322, 373)
}

terrainColor = {
    'HILLS'    : (255, 128,  0),
    'MOUNTAINS': (96, 96, 96),
    'PASTURE'  : (102, 204, 0),
    'FIELDS'   : (253, 255, 0),
    'FOREST'   : (0 , 102,  0),
    'SEA'      : (99 , 197, 249),
    'DESERT'   : (247, 210, 101),
}

portColor = {
    '3for1'        : (0, 0, 0),
    'BrickHarbor'  : (255, 128,  0),
    'OreHarbor'    : (96, 96, 96),
    'WoolHarbor'   : (102, 204, 0),
    'GrainHarbor'  : (253, 255, 0),
    'LumberHarbor' : (0 , 102,  0)
}

playerColor = [
    (249,  65,  52), # red
    ( 26,  58, 242), # blue
    (237,  14, 199), # pink
    (131, 255,  22)  # yellowish
]

def GetGameStateImage(gameState, action=None):

    if gameState is not None:

        board = Image.open("BoardArt/frame.png")
        mainImg = Image.new('RGBA', (900, 407))#board.size)
        mainImg.paste(board)

        hexArt = Image.open("BoardArt/hex.png")
        hexImg = Image.new('RGBA', hexArt.size)
        hexImg.paste(hexArt)

        settlementArt = Image.open("BoardArt/settlement.png")
        settlementImg = Image.new('RGBA', settlementArt.size)
        settlementImg.paste(settlementArt)

        cityArt = Image.open("BoardArt/city.png")
        cityImg = Image.new('RGBA', cityArt.size)
        cityImg.paste(cityArt)

        roadStraightArt = Image.open("BoardArt/road_straight.png")
        roadStraightImg = Image.new('RGBA', roadStraightArt.size)
        roadStraightImg.paste(roadStraightArt)

        roadLeftArt = Image.open("BoardArt/road_left.png")
        roadLeftImg = Image.new('RGBA', roadLeftArt.size)
        roadLeftImg.paste(roadLeftArt)

        roadRightArt = Image.open("BoardArt/road_right.png")
        roadRightImg = Image.new('RGBA', roadRightArt.size)
        roadRightImg.paste(roadRightArt)

        portPos = [0x17, 0x5b, 0x9d, 0x13, 0xdd, 0x31, 0xd9, 0x71, 0xb5]

        for boardHexIndex, boardHex in gameState.boardHexes.items():
            coloredHex = tintImage(hexImg, terrainColor[boardHex.terrain])

            if boardHexIndex in portPos:
                # print(boardHexIndex)
                for adjacentNode in boardHex.adjacentNodes:
                    node = gameState.boardNodes[adjacentNode]
                    # print(node.portType)
                    if node.portType is not None:
                        coloredHex = tintImage(hexImg, portColor[node.portType])
                        break

            hexImgPos = (int(hexPositions[boardHexIndex][0] - hexImg.size[0] / 2),
                         int(hexPositions[boardHexIndex][1] - hexImg.size[1] / 2))
            mainImg.paste(coloredHex, hexImgPos, hexImg)

            if boardHex.number is not None and boardHex.number > 0:

                draw = ImageDraw.Draw(mainImg)
                font = ImageFont.truetype("CatanData/Menlo.ttf", 25)

                x = hexPositions[boardHexIndex][0]
                y = hexPositions[boardHexIndex][1] - 15

                if boardHex.number == 11:
                    x -= 16
                elif boardHex.number >= 10:
                    x -= 15
                else:
                    x -= 9

                draw.text((x + 2, y),     "{0}".format(boardHex.number), (255, 255, 255), font=font)
                draw.text((x, y + 2),     "{0}".format(boardHex.number), (255, 255, 255), font=font)
                draw.text((x + 1, y + 2), "{0}".format(boardHex.number), (255, 255, 255), font=font)
                draw.text((x + 1, y + 1), "{0}".format(boardHex.number), (255, 255, 255), font=font)
                draw.text((x + 2, y + 1), "{0}".format(boardHex.number), (0, 0, 0), font=font)

        for playerIndex in range(0, len(gameState.players)):

            player = gameState.players[playerIndex]

            coloredStraightRoad = tintImage(roadStraightImg, playerColor[player.seatNumber])
            coloredLeftRoad = tintImage(roadLeftImg, playerColor[player.seatNumber])
            coloredRightRoad = tintImage(roadRightImg, playerColor[player.seatNumber])

            for road in player.roads:

                if road in straightRoadsPos:

                    roadImgPos = (int(straightRoadsPos[road][0] - roadStraightImg.size[0] / 2),
                                  int(straightRoadsPos[road][1] - roadStraightImg.size[1] / 2))
                    mainImg.paste(coloredStraightRoad, roadImgPos, roadStraightImg)

                elif road in leftRoadsPos:

                    roadImgPos = (int(leftRoadsPos[road][0] - roadLeftImg.size[0] / 2),
                                  int(leftRoadsPos[road][1] - roadLeftImg.size[1] / 2))
                    mainImg.paste(coloredLeftRoad, roadImgPos, roadLeftImg)

                elif road in rightRoadsPos:

                    roadImgPos = (int(rightRoadsPos[road][0] - roadRightImg.size[0] / 2),
                                  int(rightRoadsPos[road][1] - roadRightImg.size[1] / 2))
                    mainImg.paste(coloredRightRoad, roadImgPos, roadRightImg)

        for playerIndex in range(0, len(gameState.players)):

            player = gameState.players[playerIndex]

            coloredSettlement = tintImage(settlementImg, playerColor[player.seatNumber])
            coloredCity = tintImage(cityImg, playerColor[player.seatNumber])

            for settlement in player.settlements:
                settlementImgPos = (int(citySettlementPos[settlement][0] - settlementImg.size[0] / 2),
                                    int(citySettlementPos[settlement][1] - settlementImg.size[1] / 2))
                mainImg.paste(coloredSettlement, settlementImgPos, settlementImg)

            for city in player.cities:
                cityImgPos = (int(citySettlementPos[city][0] - cityImg.size[0] / 2),
                              int(citySettlementPos[city][1] - cityImg.size[1] / 2))
                mainImg.paste(coloredCity, cityImgPos, cityImg)

        draw = ImageDraw.Draw(mainImg)
        font = ImageFont.truetype("CatanData/Menlo.ttf", 20)

        res = gameState.players[0].resources[:5]

        # Brick, ore, wool, wheat, wood
        robberHex = gameState.boardHexes[gameState.robberPos]
        draw.text((470, 0), f"RobberPos: {robberHex.number}-{robberHex.production}", (255, 255, 255), font=font)
        draw.text((470, 30), f"Longest Road: Player {gameState.longestRoadPlayer}", (255, 255, 255), font=font)
        draw.text((470, 60), f"Largest Army: Player {gameState.largestArmyPlayer}", (255, 255, 255), font=font)
        draw.text((470, 90), f"VictoryPoints: {gameState.players[0].victoryPoints}", (255, 255, 255), font=font)
        draw.text((470, 120), f"            B  O  S  W  L", (255, 255, 255), font=font)
        draw.text((470, 150), f"Resources: {res}", (255, 255, 255), font=font)
        draw.text((470, 180), f"DevCards: {gameState.players[0].developmentCards}", (255, 255, 255), font=font)
        # draw.text((470, 210), f"DevCardVps: {gameState.players[0].developmentCards[VICTORY_POINT_CARD_INDEX]}", (255, 255, 255), font=font)
        # draw.text((470, 240), f"SetupProduction: {gameState.players[0].stats.setupResourceProduction[:5]}", (255, 255, 255), font=font)
        if action:
            draw.text((470, 210), f"Action: {action.type}", (255, 255, 255), font=font)
            if action.type == "AcceptTradeOffer":
                draw.text((500, 240), f"          B  O  S  W  L", (255, 255, 255), font=font)
                draw.text((500, 270), f"Give:    {action.iGive[:5]}\nReceive: {action.iGet[:5]}", (255, 255, 255), font=font)
            elif action.type == "MakeTradeOffer":
                draw.text((500, 240), f"          B  O  S  W  L", (255, 255, 255), font=font)
                draw.text((500, 270), f"Give:    {action.giveResources[:5]}\nReceive: {action.getResources[:5]}", (255, 255, 255), font=font)

        mainImg.convert('RGB')

        return mainImg

    else:
        print("ERROR! gameState is NONE!")

        return None

def SaveGameStateImage(imgFileName, gameState, action=None):

    gameStateImg = GetGameStateImage(gameState, action)

    if gameStateImg is not None:
        gameStateImg.save(imgFileName)

def DisplayImage(gameState, actionString=None):
    gameStateImg: Image.Image = GetGameStateImage(gameState, actionString)
    gameStateImg.show()

if __name__ == '__main__':

    Tk().withdraw()
    filename = askopenfilename(filetypes=(("Pickle files", "*.pickle"),
                                          ("All files", "*.*")))

    savedGameState = None

    with open('{0}'.format(filename), 'rb') as handle:
        savedGameState = pickle.load(handle)

    if savedGameState is not None:

        savedGameState.longestRoadPlayer = -1

        savedGameState.UpdateLongestRoad()

        logging.critical("#########################################################")

        logging.critical("Game Over! Player {0} Wins!".format(savedGameState.players[savedGameState.winner].name))

        logging.critical("GAME STATS:")

        logging.critical(" largest army player: {0} \n longest road player: {1} ".format(
            savedGameState.largestArmyPlayer,
            savedGameState.longestRoadPlayer
        ))

        logging.critical("#########################################################")

        for i in range(0, 4):

            logging.critical("Player {0} stats:".format(savedGameState.players[i].name))

            logging.critical("his resources are: "
                             "\n POINTS       = {0} "
                             "\n LARGEST ARMY = {1} "
                             "\n LONGEST ROAD = {2}"
                             "\n RESOURCES    = {3} "
                             "\n PIECES       = {4} "
                             "\n KNIGHTS      = {5} ".format(
                savedGameState.players[i].GetVictoryPoints(),
                savedGameState.players[i].biggestArmy,
                savedGameState.players[i].biggestRoad,
                savedGameState.players[i].resources,
                savedGameState.players[i].numberOfPieces,
                savedGameState.players[i].knights
            ))

            devCards = ""

            for j in range(0, len(g_developmentCards)):
                devCards += " {0} : {1}".format(
                    g_developmentCards[j], savedGameState.players[i].developmentCards[j]
                )

            logging.critical(" DevCards : {0}".format(devCards))

            logging.critical(" Roads: {0}\n Settlements: {1}\n Cities: {2}".format(
                [hex(road) for road in savedGameState.players[i].roads],
                [hex(settlement) for settlement in savedGameState.players[i].settlements],
                [hex(city) for city in savedGameState.players[i].cities]
            ))

            logging.critical("---------------------------------------------------------")

    gameStateImage = GetGameStateImage(savedGameState)

    if gameStateImage is not None:
        gameStateImage.show()