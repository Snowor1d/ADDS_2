from mesa import Model
from mesa import Agent
from agent import RobotAgent
from agent import CrowdAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import os 
import agent
from agent import WallAgent
import random
import copy
import math
import numpy as np
import os
import time

import agent
import model
import time
import sys

import faulthandler
faulthandler.enable()


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#-------------------------#
visualization_mode = 'off' # choose your visualization mode 'on / off
run_iteration = 1500
number_of_agents = 50 # agents 수
max_step_num = 1500
#-------------------------#

def one_hot_encode_map(state_2d, num_classes=5):
    """
    state_2d: (height, width) = (70, 70)
    num_classes: 5 (0: empty, 1: person, 2: robot, 3: wall, 4: exit)
    return: one_hot_map of shape (num_classes, height, width)
    """
    # state_2d를 LongTensor로 변환
    state_tensor = torch.LongTensor(state_2d)

    # (height, width)에 대해 one-hot → (height, width, num_classes)
    one_hot = F.one_hot(state_tensor, num_classes=num_classes)

    # 채널 차원(num_classes)을 앞으로 가져와 (num_classes, height, width)로 변환
    one_hot = one_hot.permute(2, 0, 1).float()

    return one_hot


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_directions):
        super(ActorCritic, self).__init__()

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(5, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        # Fully connected layers
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 256)

        # Actor output layers
        self.actor_direction = nn.Linear(256, num_directions)  # For movement direction
        self.actor_mode = nn.Linear(256, 2)  # Guide mode or not guide mode

        # Critic output layer
        self.critic = nn.Linear(256, 1)  # State value

    def _get_conv_out(self, shape):
        o = torch.zeros(1, 5, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        direction_probs = F.softmax(self.actor_direction(x), dim=-1)
        mode_probs = F.softmax(self.actor_mode(x), dim=-1)
        value = self.critic(x)

        return direction_probs, mode_probs, value
# Environment Interaction and Training
class TDActorCriticAgent:
    def __init__(self, input_shape, num_directions, lr=1e-4, gamma=0.99, start_epsilon=1.0):
        self.model = ActorCritic(input_shape, num_directions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.directions = ["UP", "DOWN", "LEFT", "RIGHT"]  # Movement directions
        self.modes = ["GUIDE", "NOT_GUIDE"]  # Guide modes
        epsilon_start = start_epsilon
        epsilon_min = 0.01
        epsilon_decay = 0.95
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def update_epsilon(self, is_down, decay_value):
        if (is_down):
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_value)
        else:
            self.epsilon = min(1.0, self.epsilon / decay_value)

    def select_action(self, state):
        one_hot_state = one_hot_encode_map(state)
        input_tensor = one_hot_state.unsqueeze(0)

        if np.random.rand() < self.epsilon:
            direction = np.random.choice(self.directions)
            mode = np.random.choice(self.modes)
            return direction, mode

        with torch.no_grad():
            logits_direction, logits_mode, _ = self.model(input_tensor)

            # Normalize logits to prevent overflow
            logits_direction = logits_direction - torch.max(logits_direction)
            logits_mode = logits_mode - torch.max(logits_mode)

            # Calculate probabilities
            direction_probs = F.softmax(logits_direction, dim=-1)
            mode_probs = F.softmax(logits_mode, dim=-1)

            if torch.isnan(direction_probs).any():
                print("NaN detected in direction_probs!")
                print("Logits Direction:", logits_direction)
            if (direction_probs < 0).any():
                print("Negative probabilities detected in direction_probs!")

            #print(direction_probs)
            direction_idx = torch.multinomial(direction_probs, 1).item()
            mode_idx = torch.multinomial(mode_probs, 1).item()

            direction = self.directions[direction_idx]
            mode = self.modes[mode_idx]

            return direction, mode

    def update(self, state, action, reward, next_state, done):
        
        state_tensor = one_hot_encode_map(state, num_classes=5).unsqueeze(0)
        next_state_tensor = one_hot_encode_map(next_state, num_classes=5).unsqueeze(0)

        direction_probs, mode_probs, value = self.model(state_tensor)
        _, _, next_value = self.model(next_state_tensor)

        # Compute target and advantage
        target = reward + (1 - done) * self.gamma * next_value.item()
        advantage = target - value.item()

        # Convert action to indices
        direction_idx = self.directions.index(action[0])
        mode_idx = self.modes.index(action[1])

        # Compute losses
        direction_loss = -torch.log(direction_probs[0, direction_idx]) * advantage
        mode_loss = -torch.log(mode_probs[0, mode_idx]) * advantage
        value_loss = F.mse_loss(value, torch.tensor([target]))

        loss = direction_loss + mode_loss + value_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 0.5)
        self.optimizer.step()
    

    def load_model(self, filepath):
        """Load the model and optimizer states."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {filepath}")
        else:
            print(f"No checkpoint found at {filepath}")
        
    def save_model(self, filepath):
        """Save the model and optimizer states."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def reset(self):
        """Reset agent state if necessary when starting a new simulation."""
        # Placeholder for any additional state reset logic if needed
        pass

input_shape = (70, 70)
num_directions =4 

agent = TDActorCriticAgent(input_shape, num_directions, start_epsilon = 1.0)
agent.load_model("actor_critic_model.pth")
for j in range(run_iteration):
    print(f"{j} 번째 학습 ")
    result = []
    the_number_of_model = 0

    #for model_num in range(5):
    for model_num in range(4):

        reference_step = 0
        for each_model_learning in range(1):
        # 모델 생성 및 실행에 실패하면 반복해서 다시 시도
            step_num = 0
            while True:     
                try:

                        # model 객체 생성
                    model_o = model.FightingModel(number_of_agents, 70, 70, model_num+1, 'Q')
                    the_number_of_model += 1
                    print("------------------------------")
                    print(f"{the_number_of_model}번째 학습")
                    break  # 객체가 성공적으로 생성되면 루프 탈출
                except Exception as e:
                    print(e)
                    print("error 발생, 다시 시작합니다")
                    continue  # 모델 생성에 실패하면 다시 시도
                
                # 모델이 성공적으로 생성되었으므로 step 진행
            initialized = 0
            state= model_o.return_current_image()
            state = np.array(state)
            total_reward = 0
            while True:
                #try:
                step_num += 1
                model_o.step()
                reward = 0

                action = agent.select_action(state)
                model_o.robot.receive_action(action)
                print(f"action : {action}")

                next_state = model_o.return_current_image()

                done=False
                reward = model_o.check_reward_danger() / 10
                total_reward += reward
                print(f"reward : {reward}")
                if step_num >= max_step_num or model_o.alived_agents() <= 1:
                    done= True
        
                agent.update(state, action, reward, next_state, done)
                state = next_state


                if (done):
                    break

                # except Exception as e:
                #     print(e)
                #     print("error 발생, 다시 시작합니다")
                #     # step 수행 중 오류가 발생하면, model 생성부터 다시 시작
                #     break
            del model_o
            
            decay_value = 0.95
            if(agent.epsilon < 0.1):
                decay_value = 0.995
                

            if (total_reward == 0):
                agent.update_epsilon(False, decay_value)
            else:
                agent.update_epsilon(True, decay_value)

            print("99% 탈출에 걸리는 step : ", step_num)
        print(f"현재 epsilon : {agent.epsilon}")
        save_path = 'actor_critic_model.pth'
        agent.save_model(save_path)
        print(f"model saved in {save_path}")
