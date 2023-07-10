'''main.py'''
from collections import  namedtuple
from snake_game import SnakeGameAI
from plot_training import plot_training
from agent import Agent
import logging

logging.basicConfig(filename='game.log', level=logging.INFO)

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward', 'game_over'))


def update_record(agent, score, record):
    '''Updates the game record and saves the model if necessary'''
    if score > record:
        record = score
        agent.model.save()
    return record

def log_game_info(agent, score, record):
    '''Logs the game information'''
    log = f'Game: {agent.n_games}, Score: {score}, Record: {record}'
    logging.info(log)
    print(log)

def update_plots(agent, score, total_score, plot_scores, plot_mean_scores, plot_epsilon):
    '''Updates the plots'''
    plot_epsilon.append(agent.epsilon)
    plot_scores.append(score)
    total_score += score
    mean_score = total_score / agent.n_games
    plot_mean_scores.append(mean_score)
    if agent.n_games == 1 or agent.n_games % 5 == 0:
        plot_training(plot_scores, plot_mean_scores, plot_epsilon)
    return total_score

def handle_end_of_game(agent, game, score, record, plot_scores, plot_mean_scores, plot_epsilon, total_score):
    '''Handles the end of a game'''
    game.reset()
    agent.n_games += 1
    agent.train_long_memory()

    record = update_record(agent, score, record)
    log_game_info(agent, score, record)
    total_score = update_plots(agent, score, total_score, plot_scores, plot_mean_scores, plot_epsilon)

    return record, total_score

def train():
    '''main function for training the agent'''
    plot_scores = []
    plot_mean_scores = []
    plot_epsilon = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        experience = Experience(state_old, final_move, reward, state_new, game_over)

        # train short memory
        agent.trainer.train_step(*experience)

        # remember
        agent.memory.append(experience)

        if game_over: 
            record, total_score = handle_end_of_game(agent, game, score, record, plot_scores, plot_mean_scores, plot_epsilon, total_score)

if __name__ == '__main__':
    train()