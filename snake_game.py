'''snake_game.py'''
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT, LEFT, UP, DOWN = 1, 2, 3, 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
COLLISION_REWARD = -10
FOOD_REWARD = 10
BLOCK_SIZE = 10

SPEED = 10000

class SnakeGameAI:
    '''The Snake Game AI class'''
    
    def __init__(self, w=640, h=480):
        '''Initializes the game'''
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        '''Resets the game'''
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def play_step(self, action):
        '''Plays one step of the game'''
        self.frame_iteration += 1
        self._handle_user_input()

        reward, game_over = self._move_and_check_collision(action)

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        '''Checks if the snake has collided with the boundary or itself'''
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False

    def _place_food(self):
        '''Places the food at a random point. If the food is placed on the snake, it is placed again'''
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def _handle_user_input(self):
        '''Handles the user input'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # check for pause
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self._pause()

    def _move_and_check_collision(self, action):
        '''Moves the snake and checks for collision'''
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = COLLISION_REWARD
            return reward, game_over
            
        # place new food or move
        if self.head == self.food:
            self.score += 1
            reward = FOOD_REWARD
            self._place_food()
        else:
            self.snake.pop()

        return reward, game_over
        
    
    def _update_ui(self):
        '''updates the display and draws the snake'''
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+0.2*BLOCK_SIZE, pt.y+0.2*BLOCK_SIZE, 0.6*BLOCK_SIZE, 0.6*BLOCK_SIZE))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def _calculate_new_direction(self, action):
        '''Calculates the new direction based on the action given'''

        clockwise = {Direction.RIGHT: Direction.DOWN, 
                    Direction.DOWN: Direction.LEFT, 
                    Direction.LEFT: Direction.UP, 
                    Direction.UP: Direction.RIGHT}

        anticlockwise = {Direction.RIGHT: Direction.UP, 
                        Direction.UP: Direction.LEFT, 
                        Direction.LEFT: Direction.DOWN, 
                        Direction.DOWN: Direction.RIGHT}

        if action == [1, 0, 0]:  # Straight
            pass  # no change
        elif action == [0, 1, 0]:  # Right turn
            self.direction = clockwise[self.direction]
        else:  # Left turn
            self.direction = anticlockwise[self.direction]
        
    def _move(self, action):
        '''Moves the snake based on the action given'''
        self._calculate_new_direction(action)

        direction_moves = {Direction.RIGHT: (BLOCK_SIZE, 0),
                        Direction.LEFT: (-BLOCK_SIZE, 0),
                        Direction.UP: (0, -BLOCK_SIZE),
                        Direction.DOWN: (0, BLOCK_SIZE)}

        dx, dy = direction_moves[self.direction]
        self.head = Point(self.head.x + dx, self.head.y + dy)

    def _pause(self):
        '''Pauses the game until the user presses space or p'''
        paused = True
        while paused:
            text = font.render("Paused - Press Space to Continue", True, WHITE)
            # display pause text centered
            text_rect = text.get_rect(center=(self.w/2, self.h/2))
            self.display.blit(text, text_rect)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_p:
                        paused = False
                        # remove pause text
                        text = font.render("Paused - Press Space to Continue", True, BLACK)
                        self.display.blit(text, text_rect)
                        pygame.display.flip()                   
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        
