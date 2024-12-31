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

#-------------------------#
visualization_mode = 'off' # choose your visualization mode 'on / off
run_iteration = 1500
number_of_agents = 30 # agents 수
max_step_num = 2000
#-------------------------#

previous_best_record = [4000, 4000, 4000, 4000, 4000]
reference_record = [600, 550, 1400, 1100, 800]

reference_reward = [[0]*21, [0]*21, [0]*21, [0]*21, [0]*21]
#[[0]]
#reference_reward dict list 만들기

for i in range(4):
    for j in range(3):
        step = 0
        model_o1 = model.FightingModel(number_of_agents, 70, 70, i+1, 'Q')
        while(True):
            try:
                step += 1
                model_o1.step()
                if(step%100 == 0):
                    reference_reward[i][int(step/100)] += model_o1.evacuated_agents()
                if (model_o1.alived_agents() <= 3 or step>=max_step_num):
                    for step_num in range(int(step/100)+1, 21):
                        reference_reward[i][step_num] += model_o1.evacuated_agents()
                    break
            except Exception as e:
                print(e)
                print("error 발생, 다시 시작합니다")
                continue  # 모델 생성에 실패하면 다시 시도
        del model_o1
    print(f"{i+1}번째 모델의 reference_reward 생성함")

for i in range(5):
    for j in range(20):
        reference_reward[i][j] = reference_reward[i][j]/3

print("reference_reward 생성 완료")
print(reference_reward)



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
            while True:
                try:
                    step_num += 1
                    model_o.step()
                    #reward = (model_o.check_reward(reference_reward[model_num])+1.5)/100
                    reward = 0
                    #reward += model_o.check_reward_danger() / 1000
                    # if(reward == 0):
                    #   reward = -0.01 
                    if(step_num%20 == 0):
                        if (initialized != 0):
                          reward = model_o.evacuated_agents()-reference_reward[model_num][int(step_num/100)]
                          if(reward > 1):
                            reference_reward[model_num][int(step_num/100)] += 0.1
                          print("reward : ", reward)
                          model_o.robot.update_weight(reward)
                        else:
                          reward = 0
                          initialized = 1
 
                    if step_num >= max_step_num:
                        break
                    if model_o.alived_agents() <= 1:
                        break
                except Exception as e:
                    print(e)
                    print("error 발생, 다시 시작합니다")
                    # step 수행 중 오류가 발생하면, model 생성부터 다시 시작
                    break
            del model_o


            print("99% 탈출에 걸리는 step : ", step_num)