# Snake Game AI

This repository contains an implementation of a Snake game AI. The AI agent is trained using a Deep Q-Learning model to maximize its score in the game.

## Overview
The project is organized as follows:
1. `snake_game.py`: Contains the implementation of the Snake game.
2. `agent.py`: Defines the agent that plays the game. The agent uses a Deep Q-Learning model to decide the best action given the current game state.
3. `model.py`: Defines the Q-Learning model and the training process.
4. `main.py`: The main entry point of the program. Starts the game and the training process.
5. `plot_training.py`: Defines a function to plot the score, mean score, and epsilon value versus the number of games played. This allows you to track performance during training.

## Installation
Before running this project, ensure that you have installed all the necessary packages listed in the `requirements.txt` file. If you haven't, you can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage
To start the game and the training process, run:

```bash
python main.py
```

## How it Works
The agent uses a Deep Q-Learning model to learn how to play the game. At each step, the agent receives the game state (a 12-dimensional vector) and returns an action to perform. The game state consists of the following information:

- Danger in four directions (left, right, up, down)
- Current direction of the snake (left, right, up, down)
- Relative position of the food (left, right, up, down)

The agent uses an epsilon-greedy strategy for action selection. This means that with a probability of `epsilon`, the agent will select a random action (exploration), and with a probability of `1 - epsilon`, the agent will select the action that it believes will yield the maximum future reward (exploitation). The value of `epsilon` starts at 1 and decays exponentially over time, causing the agent to favor exploitation over exploration as it learns more about the environment.

The agent is trained using a batch of experiences from its replay memory. Each experience consists of the current state, the action performed, the reward received, the next state, and whether the game is over. The agent uses the Bellman equation to update its Q-value estimates, and the weights of the neural network are updated using backpropagation.

## Acknowledgements
This project was inspired by [Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds&t=1727s). This tutorial provided a great starting point for implementing a Deep Q-Learning model in PyTorch. The repo for the video can be found here: [snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch)