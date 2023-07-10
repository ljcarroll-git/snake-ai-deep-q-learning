'''plot_training.py'''
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

def plot_training(scores, mean_scores, epsilon_history):
    '''Plots the training scores, mean scores and epsilon history'''
    ax1.clear()
    ax2.clear()

    # Plot scores and mean scores
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Score')
    ax1.plot(scores, 'b-')
    ax1.plot(mean_scores, 'r-')
    ax1.tick_params(axis='y')
    ax1.legend(['Score', 'Mean Score'], loc='upper left')

    # Plot epsilon history
    ax2.set_ylabel('Epsilon')
    ax2.plot(epsilon_history, 'g-')
    ax2.tick_params(axis='y')
    ax2.legend(['Epsilon'], loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training')
    plt.draw()
    plt.pause(0.01)