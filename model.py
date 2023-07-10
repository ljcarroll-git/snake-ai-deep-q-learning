'''model.py'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    '''Linear Q Network'''
    def __init__(self, input_size, hidden_size1, output_size):
        '''Initializes the network'''
        super().__init__() # super class
        self.linear1 = nn.Linear(input_size, hidden_size1) # input layer
        self.linear2 = nn.Linear(hidden_size1, output_size)

    def forward(self, x): 
        '''Does the forward pass'''
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        '''Saves the model'''
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        # check if file exists rename it if it does
        file_name = os.path.join(model_folder_path, file_name)
        # print(file_name)
        # while os.path.isfile(file_name):
        #     file_name = file_name.split('.')[0] + '_new.' + file_name.split('.')[1]
     
        torch.save(self.state_dict(), file_name) # save model


class QTrainer:
    '''QTrainer'''
    def __init__(self, model, lr, gamma):
        '''Initializes the QTrainer'''
        self.lr = lr # learning rate
        self.gamma = gamma # discount rate
        self.model = model # model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # optimizer
        self.criterion = nn.MSELoss() # loss function

    def train_step(self, state, action, reward, next_state, done): # train model
        '''Trains the model using the Q-learning algorithm (Bellman Equation)'''
        state = torch.tensor(state, dtype=torch.float) # convert to tensor
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: # if state is 1D, convert to 2D
            # unsqueeze(0) adds a dimension of size 1 at the specified position
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # convert to tuple

        # predicted Q values with current state (Q(s, a, w))
        pred = self.model(state)

        target = pred.clone() # clone the predicted Q values
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action).item()] = Q_new


        # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone() -> clone the predicted Q values
        # preds[argmax(action)] = Q_new -> update the predicted Q values
        self.optimizer.zero_grad() # reset the optimizer
        loss = self.criterion(target, pred) # calculate the loss
        loss.backward() # back propagation
        self.optimizer.step() # update the weights



