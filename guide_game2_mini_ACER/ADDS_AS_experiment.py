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
run_iteration = 3
number_of_agents = 30 # agents 수
max_step_num = 3000
#-------------------------#

adds_benchmark = 0

for j in range(run_iteration):
    print(f"{j} 번째 학습 ")
    result = []
    the_number_of_model = 0

    for model_num in range(5):

        reference_step = 0
        for each_model_learning in range(1):
        # 모델 생성 및 실행에 실패하면 반복해서 다시 시도
            step_num = 0
            while True:
                try:
                        # model 객체 생성
                    model_o = model.FightingModel(number_of_agents, 40, 40, model_num+1, 'Q')
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
                model_o.step()
                step_num += 1
                try:
                    if model_o.alived_agents() <= 3:
                        adds_benchmark += step_num
                        break
                    if step_num > 2000:
                        adds_benchmark += step_num
                        break
                except Exception as e:
                    print(e)
                    print("error 발생, 다시 시작합니다")
                    # step 수행 중 오류가 발생하면, model 생성부터 다시 시작
                    break
            del model_o


            print("탈출에 걸리는 step : ", step_num)

with open("benchmark_score.txt", "w") as file:
    file.write(f"adds_benchmark: {adds_benchmark}\n")

print("benchmark_score.txt 파일에 값이 저장되었습니다.")