import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pickle
from CatanSimulator import CreateGame
from Game.CatanGame import *
from Agents.AgentRandom2 import AgentRandom2
from DeepLearning.GetObservation import lowerBoundsSimplified, upperBoundsSimplified, getObservationSimplified
from DeepLearning.GetActionMask import getActionMask



class CityStrategyEnv(gym.Env):
    """
    Full Catan game with full action and state space (no player trades)
    Rewards: BuildCity - 20, BuildSettlement(First 2) - 8, BuyDevelopementCard - 5, PlayDevCard - 1, LargestArmy/Road - 10, win/loss - +40/-40
    SetupRewards: Stone - 2xdots/-20, Wheet- 1.5*dots/-20, Sheep - dots/-5
    """
    def __init__(self, customBoard=None):
        super(CityStrategyEnv, self).__init__()

        self.action_space = spaces.Discrete(486)
        self.observation_space = spaces.Box(low=lowerBoundsSimplified, high=upperBoundsSimplified, dtype=np.int64) 

        self.game: Game = None
        # Used to fetch Action object from action index chosen by models
        self.indexActionDict = None
        self.players = []
        self.agent = None
        self.customBoard = customBoard
        self.num_envs = 1
        self.action_mask = None
        self.seed = 1

        self.getActionMask = getActionMask
        self.getObservation = getObservationSimplified

    def reset(self, seed=None):
        """
        Setup new game, cycle through actions till agents turn, return current state and set action mask
        """
        inGame = CreateGame([   AgentRandom2("P0", 0),
                                AgentRandom2("P1", 1),
                                AgentRandom2("P2", 2),
                                AgentRandom2("P3", 3)], self.customBoard)
        self.game = pickle.loads(pickle.dumps(inGame, -1))
        self.players = self.game.gameState.players
        self.agent = self.game.gameState.players[0]

        # Cycle through until agents turn
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]

        # Return initial info needed: State, ActionMask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        return observation, {}


    def step(self, action):
        """
        Accepts action index as argument, applies action, cycles through to players next turn, 
        gets observation and action mask for turn
        """
        truncated = False
        done = False

        reward = 0
        biggestArmyBefore = self.agent.biggestArmy
        biggestRoadBefore = self.agent.biggestRoad
        prevState = self.game.gameState.currState

        # Apply action chosen
        actionObj = self.indexActionDict[action]
        actionObj.ApplyAction(self.game.gameState)

        # get reward
        if biggestArmyBefore == False and self.agent.biggestArmy == True:
            reward += 20
        if biggestRoadBefore == False and self.agent.biggestRoad == True:  
            reward += 20
        if actionObj.type == 'BuildSettlement' and self.game.gameState.currState[:5] != 'START' and len(self.agent.settlements) <= 4:
            reward += 8
        elif actionObj.type == 'BuildCity':
            reward += 20
        elif actionObj.type == 'BuyDevelopmentCard':
            reward += 5
        elif actionObj.type == 'UseKnightsCard' or actionObj.type == 'UseMonopolyCard' or actionObj.type == 'UseYearOfPlentyCard' or actionObj.type == 'UseFreeRoadsCard':
            reward += 1
        elif actionObj.type == 'BuildSettlement' and prevState == 'START2A':
            resourceProduction = listm([0, 0, 0, 0, 0, 0])
            for diceNumber, resourceList in self.game.gameState.players[0].diceProduction.items():
                resourceProduction += [numberDotsMapping[diceNumber] * resource for resource in resourceList]
            ore = resourceProduction[1]
            wool = resourceProduction[2]
            wheat = resourceProduction[3]
            if ore <= 0:
                reward-= 20
            else:
                reward += ore * 2
            if wheat <= 0:
                reward -= 20
            else:
                reward += wheat * 1.5
            if wool > 0:
                reward += wool


        if self.game.gameState.currState == "OVER":
            wonGame = self.game.gameState.winner == 0
            if wonGame:
                reward += 40
            else:
                reward -= 40
            return None, reward, True, truncated, {}
        
        # if game is not over cycle through actions until its agents turn again
        currPlayer = self.players[self.game.gameState.currPlayer]
        while currPlayer.seatNumber != 0:
            agentAction = currPlayer.DoMove(self.game)
            agentAction.ApplyAction(self.game.gameState)
            currPlayer = self.players[self.game.gameState.currPlayer]
            if self.game.gameState.currState == "OVER":
                wonGame = self.game.gameState.winner == 0
                if wonGame:
                    reward += 40
                else:
                    reward -= 40
                return None, reward, True, truncated, {}
        
        # Now ready for agent to choose action, get observation and action mask
        possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        if len(possibleActions) == 1 and possibleActions[0].type == "ChangeGameState":
            possibleActions[0].ApplyAction(self.game.gameState)
            possibleActions = self.agent.GetPossibleActions(self.game.gameState)
        self.action_mask, self.indexActionDict = self.getActionMask(possibleActions)
        observation = self.getObservation(self.game.gameState)

        # observation, reward, terminated, truncated, info
        return observation, reward, done, truncated, {}
    
    # Used by PPO algorithm
    def action_masks(self):
        return self.action_mask