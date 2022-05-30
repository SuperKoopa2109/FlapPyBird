import flappy
import itertools
import numpy as np
from numpy.linalg import norm
import sys

class Agent_Q_FLAP():
    Q = [] # Q Values
    A = {} # Actions
    S = {} # States
    R = {} # Rewards
    discount = 1 # discount
    learning_rate = 0.1 # This is the variable for the learning rate.
    exploration = 0.05
    continuous = False
    showUI = True
    max_score = 0

    def __init__(self, R, showUI = True, continuous = False):
        # initialize Qs
        self.R = R 
        self.A = [0, 1]
        self.S = [(x * 50, y * 50, vel) for x in range(100) for y in range(100) for vel in np.arange(-15, 16)]
        self.Q = np.random.rand(len(self.S), len(self.A))
        self.continuous = continuous
        self.showUI = showUI
        #print(self.S)
        
    def get_state(self, playerx, playery, playerVelY, upperPipes, lowerPipes):
        
        playerMidPos = playerx + flappy.IMAGES['player'][0].get_width() / 2
        dUpper = 0
        dLower = 0
        #print('playerX: ', playerx, 'playerY: ', playery)
        #print('Player Y Velocity: ', playerVelY)
        
        delta_x = 0
        delta_y = 0

        for lPipe in lowerPipes:
            # Use distance between x and y coordinates instead
            if lPipe['x'] > playerx:
                if (delta_x != 0) or (delta_y != 0):
                    break
                delta_x = round( ( lPipe['x'] - playerx ) / 50 ) * 50
                delta_y = round( ( lPipe['y'] - playery ) / 50 ) * 50
            if ( (delta_x, delta_y, playerVelY) in self.S ):
                return (delta_x, delta_y, playerVelY)
            else:
                return (0, 0, playerVelY)
        # for uPipe, lPipe in zip(upperPipes, lowerPipes):
            #if uPipe['x'] > playerx:
                #print('THE NORM ', norm(np.array([playerx, playery]), np.array([uPipe['x'], uPipe['y']])) )
                
                

                # Euclidean distance
                #dUpper_new = np.sqrt( ( uPipe['x'] - playerx ) ** 2 + ( uPipe['y'] - playery ) ** 2 )
                #dLower_new = np.sqrt( ( lPipe['x'] - playerx ) ** 2 + ( lPipe['y'] - playery ) ** 2 )
                #if dUpper_new < dUpper or dUpper == 0:
                #    dUpper = round( dUpper_new / 50 ) * 50
                #if dLower_new < dLower or dLower == 0:
                #    dLower = round( dLower_new / 50 ) * 50
        #if ( (dLower, dUpper, playerVelY) in self.S ):
        #    return (dLower, dUpper, playerVelY)
        #else:
        #    return (0, 0, playerVelY)


    def pass_action(self, state):
        
        # Exploration check
        explore_check = np.random.rand()
        if explore_check < self.exploration:
            return np.random.choice(self.A)
        else:
            return np.argmax( np.array([self.calc_Q(state, a) for a in self.A]) )

    def print_Q(self):
        print('current_Q: ', self.Q)

    def print_progress(self, score):
        if ( score > self.max_score ):
            self.max_score = score
        print('reached score: ', score, '   ', 'best score: ', self.max_score)

    def get_rewards(self):
        print(self.R)

    def calc_Q(self, state, action):
        if action == 0:
            next_state = ( state[0], state[1], state[2] + 1 )
        elif action == 1:
            next_state = ( state[0], state[1], -9 )
        else:
            next_state = (0,0,0)
        state_idx = 0
        next_state_idx = 0
        action_idx = 0
        if (state in self.S):
            state_idx = list.index(self.S, state)
        if (next_state in self.S):
            next_state_idx = list.index(self.S, next_state)
        if (action in self.A):
            action_idx = list.index(self.A, action)

        #print(self.Q[next_state_idx, :])
        #print(self.learning_rate, self.discount, np.max(self.Q[next_state_idx, :]))
        self.Q[state_idx, action_idx] = self.Q[state_idx, action_idx] + self.learning_rate * (self.R['win'] + self.discount * np.max(self.Q[next_state_idx, :]) - self.Q[state_idx, action_idx] )
        return ( self.Q[state_idx, action_idx] )

if __name__ == '__main__':
    showUI = True
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'True':
            showUI = True
        elif sys.argv[1] == 'False':
            showUI = False
    Agent_Q_FLAP = Agent_Q_FLAP(R = {'win': 1, 'lose': -10000}, showUI = showUI, continuous = True)
    Agent_Q_FLAP.get_rewards()
    flappy.main(Agent_Q_FLAP)