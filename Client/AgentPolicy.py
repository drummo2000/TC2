from CatanGame import GameState, constructableNodesList, constructableHexesList, constructableEdgesList
from CatanPlayer import Player
from CatanBoard import BoardNode, BoardHex, BoardEdge, g_portType, portTypeIndex, resourceIndex, constructionTypeIndex, numberDotsMapping
from CatanAction import *
from itertools import combinations
import math
import random
from ActionMask import getActionMask
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ActionMask import allActionsList
from AgentRandom2 import AgentRandom2
from ModelState import getInputState

class PolicyNetwork(nn.Module):
    # sets up the shape of the NN and the optimizer used to update parameters
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # Takes in the current state and returns the probabilities of taking the action
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state, action_mask):
        state = torch.from_numpy(state).float().reshape(-1).unsqueeze(0)

        probs = self.forward(state)
        # Must reduce prob of masked actions to 0
        probs = probs * torch.from_numpy(action_mask).float().reshape(-1).unsqueeze(0)

        # Normalize the masked probabilities
        sum_masked_probs = torch.sum(probs)
        probs = probs / sum_masked_probs

        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
    
    # pass the network, list of rewards, list of log(probs) of actions taken
    def update_policy(self, reward, log_probs):

        # Convert the episode return to a tensor
        return_tensor = torch.tensor(reward, dtype=torch.float)

        # get the policy gradient for each timestep
        policy_gradient = []
        for log_prob in log_probs:
            policy_gradient.append(-log_prob * return_tensor)
        
        # Reset parameter gradients
        self.optimizer.zero_grad()
        # we get the total (loss), gradient term which we must optimize our parameters around(i think the -log_prob turns it into minimization)
        policy_gradient = torch.stack(policy_gradient).sum()
        # Compute gradients of weights with respect to policy gradient term
        policy_gradient.backward()
        # optimize weights
        self.optimizer.step()


class AgentPolicy(AgentRandom2):

    def __init__(self, name, seatNumber, network: PolicyNetwork, playerTrading=True):

        super(AgentPolicy, self).__init__(name, seatNumber, playerTrading)
        self.agentName              = name
        self.network                = network
    
    # Return selected action
    def DoMove(self, game) -> (Action, list):

        # If not my turn and were not in WAITING_FOR_DISCARDS phase then return None
        if game.gameState.currPlayer != self.seatNumber and \
            game.gameState.currState != "WAITING_FOR_DISCARDS":
            return None

        possibleActions = self.GetPossibleActions(game.gameState)

        if type(possibleActions) != list:
            possibleActions = [possibleActions]

        if len(possibleActions) == 1:
            return possibleActions[0], None
        else:
            actionMask, indexActionDict = getActionMask(possibleActions)
            state = getInputState(game.gameState)

            # action is the index of selected action
            actionIndex, logProb = self.network.get_action(state, actionMask)
            return indexActionDict[actionIndex], logProb
