'''agent.py'''
import torch
import random
import numpy as np
from collections import deque, namedtuple
from snake_game import Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPS_DECAY_RATE = 0.95
EPSILON_START = 1.0

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward', 'game_over'))

class Agent:
    '''The agent that plays the game'''

    def __init__(self) -> None:
        '''Initializes the agent'''
        self.n_games = 0
        self.epsilon = 1.0 # randomness
        self.gamma = 0.9 # discount rate, closer to 1 learns well into distant future, closer to 0 learns only for current reward
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(12, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        '''
        returns the state of the game as a numpy array
        state is a 12-dimensional vector:
        [danger left, danger right, danger up, danger down,
        direction left, direction right, direction up, direction down,
        food left, food right, food up, food down]
        '''
        head = game.snake[0]
        directions = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
        points = [Point(head.x - BLOCK_SIZE, head.y),
                  Point(head.x + BLOCK_SIZE, head.y),
                  Point(head.x, head.y - BLOCK_SIZE),
                  Point(head.x, head.y + BLOCK_SIZE)]

        direction_states = [game.direction == d for d in directions]
        danger_states = [game.direction == d and game.is_collision(p) for d, p in zip(directions, points)]
        food_states = [game.food.x < game.head.x, # food left
                       game.food.x > game.head.x, # food right
                       game.food.y < game.head.y, # food up
                       game.food.y > game.head.y] # food down
        
        state = danger_states + direction_states + food_states
        

        return np.array(state, dtype=int)


    def train_long_memory(self):
        '''Trains the model on the batch of experiences in memory'''
        mini_sample = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))
        experiences = zip(*mini_sample)
        self.trainer.train_step(*experiences)

    def get_action(self, state):
        '''Returns the action to take given the current state'''
        # random moves: tradeoff exploration / exploitation
        if random.random() < self.epsilon:
            # explore
            move = random.randint(0, 2)
        else:
            # exploit
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        self.epsilon = EPSILON_START * EPS_DECAY_RATE ** self.n_games
        return final_move
