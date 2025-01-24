from mesa import Model
from mesa import Agent

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


#-------------------------#
visualization_mode = 'on' # choose your visualization mode 'on / off
run_iteration = 1500
number_of_agents = 30 # agents 수
#-------------------------#


from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import NumberInput
from model import FightingModel
from mesa.visualization.modules import CanvasGrid, ChartModule



import asyncio
import os
import platform
import webbrowser
from typing import ClassVar

import tornado.autoreload
import tornado.escape
import tornado.gen
import tornado.ioloop
import tornado.web
import tornado.websocket
from PIL import Image      
from mesa_viz_tornado.UserParam import UserParam
import cv2
most_danger = 0
width = 70
height = 70
model_num = 2
if visualization_mode == 'on':
    Width = width
    Height =height
    s_model_r = model.FightingModel(30,width,height,model_num)
    s_model_r.use_model('dqn_checkpoint_ep_590.pth')  
    ran_num = random.randint(10000,20000)
    most_danger_mesh = None
    for agent in s_model_r.schedule_e.agents:
        if agent.type == 99:
            if agent.danger > most_danger:
                most_danger = agent.danger
    0
    
    
    

    ## grid size
    #NUMBER_OF_CELLS = 50 ## square # 한 셀당 50cm x 50cm로 하겠음. 이 시뮬레이션 모델에서는 한 셀당 하나의 사람만 허용 cell 개수가 100개 -> 50m x 50m 크기의 맵
    agent_nums = 100
    SIZE_OF_CANVAS_IN_PIXELS_X = Width*20
    SIZE_OF_CANVAS_IN_PIXELS_Y = Height*20
    

    simulation_params = {
        "number_agents": NumberInput(
            "Hi, ADDS . Choose how many agents to include in the model", value=agent_nums
        ),
        "width": Width,
        "height": Height
    }

    def agent_portrayal(agent):

        global most_danger
        # if the agent is buried we put it as white, not showing it.
        if agent.buried:
            portrayal = {
                "Shape": "circle",
                "Filled": "true", ## ?
                "Color": "white", 
                "r": 0.01,
                "text": "",
                "Layer": 0,
                "text_color": "black",
            }
            return portrayal
        if agent.type == 20: ## for exit_way_rec 
            portrayal = {
                "Shape": "circle",
                "Filled": "true",
                "Color": "lightblue", 
                "r": 1,
                "text": "",
                "Layer": 0,
                "text_color": "black",
            }
            return portrayal
        if agent.type == 10: ## exit_rec 채우는 agent 
            portrayal = {
                "Shape": "circle",
                "Filled": "true",
                "Color": "green", 
                "r": 1,
                "text": "",
                "Layer": 1,
                "text_color": "black",
            }
            return portrayal
        
        if agent.type == 11: ## wall 채우는 agent 
            portrayal = {
                "Shape": "circle",
                "Filled": "true",
                "Color": "black", 
                "r": 1,
                "text": "",
                "Layer": 2,
                "text_color": "black",
            }
            return portrayal
        if agent.type == 12: ## for space visualization 
            portrayal = {
                "Shape": "circle",
                "Filled": "true",
                "Color": "lightgrey", 
                "r": 1,
                "text": "",
                "Layer": 0,
                "text_color": "black",
            }
            return portrayal

        # the default config is a circle
        portrayal = {
            
            "Shape": "circle",
            "Filled": "true",
            "r": 0.5,
            ##"text": f"{agent.health} Type: {agent.type}",
            "text_color": "black",
        }

        # if the agent is dead on the floor we change it to a black rect
        if agent.dead:
            portrayal["Shape"] = "rect"
            portrayal["w"] = 0.2
            portrayal["h"] = 0.2
            portrayal["Color"] = "black"
            portrayal["Layer"] = 11

            return portrayal
        
        portrayal["r"] = 1
        if agent.type == 0:
            portrayal["Color"] = "magenta"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 1: 
            portrayal["Color"] = f"rgb(110, 110, 250)"
            portrayal["Layer"] = 1
            return portrayal
        if agent.type == 2: 
            portrayal["Color"] = f"rgb(110, 110, 250)"
            portrayal["Layer"] = 1
            return portrayal
        if agent.type == 3: #robot
            if s_model_r.robot_mode == "GUIDE": #끌고갈때
                portrayal["Color"] = "red" #빨강!!!!!!!!!!!1
            else:
                portrayal["Color"] = "purple"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 4:
            portrayal["Color"] = "yellow"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 5:
            portrayal["Color"] = "green"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 6:
            portrayal["Color"] = "lightblue"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 7:
            portrayal["Color"] = "lightgreen"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 8:
            portrayal["Color"] = "lightgreen"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 9:
            portrayal["Color"] = "black"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 10:
            portrayal["Color"] = "white"
            portrayal["Layer"] = 2
            return portrayal
        if agent.type == 11:
            portrayal["Color"] = "grey"
            portrayal["Layer"] = 2
            return portrayal
        
        if agent.type == 99: #this for danger visualization
            red_value = int(agent.danger/most_danger*255)
            portrayal["Color"] = f"rgb(255,{(1-(agent.danger/most_danger)/3)*255},{(1-(agent.danger/most_danger)/3)*255})"
            portrayal["Layer"] = 0
            return portrayal

        if agent.type == 102:
            portrayal["Color"] = "rgb(255, 255, 255)"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 103:
            portrayal["Color"] = "rgb(240, 240, 240)"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 104:
            portrayal["Color"] = "rgb(230, 230, 230)"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 103:
            portrayal["Color"] = "rgb(220, 220, 220)"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 104:
            portrayal["Color"] = "rgb(210, 210, 210)"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 105:
            portrayal["Color"] = "rgb(200, 200, 200)"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 105:
            portrayal["Color"] = "rgb(190, 190, 190)"
            portrayal["Layer"] = 0
            return portrayal
        if agent.type == 106:
            portrayal["Color"] = "rgb(180, 180, 180)"
            portrayal["Layer"] = 0
            return portrayal

        # if agent.type == 5:
        #     portrayal["Color"] = "green"
        #     return portrayal
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 10
        return portrayal

    grid = CanvasGrid(
        agent_portrayal,
        Width,
        Height,
        SIZE_OF_CANVAS_IN_PIXELS_X,
        SIZE_OF_CANVAS_IN_PIXELS_Y
    )

    chart_healthy = ChartModule(
        [
            {"Label": "Remained Agents", "Color": "blue"},
            #{"Label": "Non Healthy Agents", "Color": "red"}, ## 그래프 상에서 Non Healthy Agents 삭제
        ],
        canvas_height = 300,
        data_collector_name = "datacollector_currents",
    )


    server2 = ModularServer(     # 이게 본체인데,,,
        FightingModel, # 내 모델
        #[grid, chart_healthy], # visualization elements 써줌
        [grid, chart_healthy],
        "ADDS crowd system", # 웹 페이지에 표시되는 이름
        simulation_params,
        8522,
        s_model_r
    )

    

    server2.port = 8522
    
    server2.launch()
    
    



