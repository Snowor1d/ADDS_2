#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Crowd Simulation (GUI)
- SIM_TIMESTEP = 0.25 s
- 자동 레이아웃: world (width, height)를 좌측 가용 화면에 '비율 유지'로 맞춰 스케일
- Pause/Resume, Step, Reset, Speed, Map ID textbox, Agents stepper
- 에피소드 종료 시 자동 재시작
- stride/downsampling 없음 (매 프레임 렌더)
"""

import os, math, time, sys
import numpy as np
import pygame

from model import FightingModel
from continuous_renderer import ContinuousRenderer

# =========================
# 화면/월드 설정
# =========================
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 900     # 창 크기
PANEL_RIGHT_WIDTH = 300                     # 우측 패널 고정 폭
PADDING = 10                                # 좌/우/상/하 패딩

# 기본 월드 크기(리셋 시 사용). 더 크게 바꿔도 화면 내에 자동으로 맞춰 그려짐
GRID_W, GRID_H = 100, 100

SIM_TIMESTEP = 0.25
TARGET_RENDER_FPS = 10
SPEED_MIN, SPEED_MAX = 0.25, 16.0
MAX_STEPS = 3000
MAX_ACCUM_STEPS = 12
MAX_SUBSTEPS_PER_FRAME_HARD = 6
ADAPTIVE_SUBSTEP_INIT = 4

ROBOT_CONTROL_MODE = "external"   # "external" | "internal"
ROBOT_VERSION_FOR_MODEL = "Q"

USE_CONTINUOUS_RENDERER = True
def make_renderer(world_w, world_h):
    return ContinuousRenderer(
        world_size=(float(world_w), float(world_h)),
        crowd_colors={0:"#4e79a7",1:"#4e79a7",2:"#4e79a7"},
        robot_color="#e15759",
        show_agent_heading=False,
        show_robot_heading=True,
        robot_heading_scale=3,
        trail_target="robot",
        trail_style="fade",
        max_trail=2000,
        single_color_edges=True,
        exit_size=5.0,
        snap_exit_to_boundary=True,
    )

WHITE=(255,255,255); BLACK=(0,0,0); GREY=(210,210,210)
DARK=(40,40,42); ACCENT=(18,136,255)

# ====== 간단 GUI 위젯 ======
class UIButton:
    def __init__(self, rect, label, on_click, font, bg=(240,240,240), fg=DARK):
        self.rect = pygame.Rect(rect); self.label = label
        self.on_click = on_click; self.font = font
        self.bg = bg; self.fg = fg; self.hover=False
    def draw(self, surf):
        color = (248,248,248) if self.hover else self.bg
        pygame.draw.rect(surf, color, self.rect, border_radius=8)
        pygame.draw.rect(surf, (200,200,200), self.rect, 1, border_radius=8)
        txt = self.font.render(self.label() if callable(self.label) else self.label, True, self.fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))
    def handle(self, e):
        if e.type==pygame.MOUSEMOTION: self.hover=self.rect.collidepoint(e.pos)
        elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.rect.collidepoint(e.pos):
            self.on_click()

class UISlider:
    def __init__(self, rect, vmin, vmax, value, on_change):
        self.rect = pygame.Rect(rect); self.vmin=vmin; self.vmax=vmax
        self.value=float(value); self.on_change=on_change; self.drag=False
    def draw(self, surf):
        track = self.rect.inflate(0,-14); track.centery=self.rect.centery
        pygame.draw.rect(surf,(230,230,230),track,border_radius=6)
        pygame.draw.rect(surf,(200,200,200),track,1,border_radius=6)
        t=(self.value-self.vmin)/(self.vmax-self.vmin)
        knob_x = track.left + int(t*track.width)
        pygame.draw.circle(surf, ACCENT, (knob_x,track.centery), 8)
        pygame.draw.circle(surf, (255,255,255), (knob_x,track.centery), 8, 2)
    def handle(self, e):
        def set_from_x(mx):
            track = self.rect.inflate(0,-14); track.centery=self.rect.centery
            t = (mx-track.left)/max(1,track.width); t=max(0,min(1,t))
            self.value = self.vmin + t*(self.vmax-self.vmin); self.on_change(self.value)
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.rect.collidepoint(e.pos):
            self.drag=True; set_from_x(e.pos[0])
        elif e.type==pygame.MOUSEBUTTONUP and e.button==1: self.drag=False
        elif e.type==pygame.MOUSEMOTION and self.drag: set_from_x(e.pos[0])

class UIStepper:
    def __init__(self,label,x,y,w,value,step,on_change,font,min_val=0):
        self.label=label; self.value=int(value); self.step=int(step)
        self.on_change=on_change; self.font=font; self.min_val=min_val
        self.rect_label=pygame.Rect(x,y,w,20)
        self.btn_minus=pygame.Rect(x,y+20,32,26)
        self.box=pygame.Rect(x+36,y+20,w-72,26)
        self.btn_plus=pygame.Rect(x+w-32,y+20,32,26)
    def draw(self,s):
        s.blit(self.font.render(self.label,True,DARK),(self.rect_label.left,self.rect_label.top))
        for btn,sym in ((self.btn_minus,"-"),(self.btn_plus,"+")):
            pygame.draw.rect(s,(242,242,242),btn,border_radius=6)
            pygame.draw.rect(s,(200,200,200),btn,1,border_radius=6)
            s.blit(self.font.render(sym,True,DARK),btn.move(10,2))
        pygame.draw.rect(s,(250,250,250),self.box,border_radius=6)
        pygame.draw.rect(s,(200,200,200),self.box,1,border_radius=6)
        s.blit(self.font.render(str(self.value),True,DARK), self.font.render(str(self.value),True,DARK).get_rect(center=self.box.center))
    def handle(self,e):
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
            if self.btn_minus.collidepoint(e.pos):
                self.value=max(self.min_val,self.value-self.step); self.on_change(self.value)
            elif self.btn_plus.collidepoint(e.pos):
                self.value=self.value+self.step; self.on_change(self.value)

class UIInputNumber:
    def __init__(self,label,x,y,w,value,on_change,font,min_val=0):
        self.label=label; self.font=font; self.text=str(int(value))
        self.rect_label=pygame.Rect(x,y,w,20); self.box=pygame.Rect(x,y+20,w,26)
        self.focus=False; self.on_change=on_change; self.min_val=min_val
        self._blink_t=0.0; self._show_caret=True
    def draw(self,s):
        s.blit(self.font.render(self.label,True,DARK),(self.rect_label.left,self.rect_label.top))
        pygame.draw.rect(s,(250,250,250),self.box,border_radius=6)
        pygame.draw.rect(s,(120,180,255) if self.focus else (200,200,200),self.box,1,border_radius=6)
        txtsurf=self.font.render(self.text,True,DARK)
        r=txtsurf.get_rect(midleft=(self.box.left+8,self.box.centery)); s.blit(txtsurf,r)
        if self.focus:
            self._blink_t+=1/60
            if self._blink_t>=0.5: self._blink_t=0.0; self._show_caret=not self._show_caret
            if self._show_caret:
                cx=r.right+2; pygame.draw.line(s,DARK,(cx,self.box.top+5),(cx,self.box.bottom-5),1)
    def handle(self,e):
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
            self.focus=self.box.collidepoint(e.pos)
        elif e.type==pygame.KEYDOWN and self.focus:
            if e.key==pygame.K_RETURN:
                if self.text.strip()=="": self.text=str(self.min_val)
                try: self.on_change(max(self.min_val,int(self.text))); 
                except ValueError: pass
                self.focus=False
            elif e.key==pygame.K_BACKSPACE:
                self.text=self.text[:-1]
                if self.text.strip()!="":
                    try: self.on_change(int(self.text))
                    except ValueError: pass
            else:
                ch=e.unicode
                if ch.isdigit():
                    self.text+=ch
                    try: self.on_change(int(self.text))
                    except ValueError: pass

# ====== 레이아웃 계산 (여기가 핵심!) ======
# 좌측 가용 영역 안에 world(width,height) 비율을 유지하며 딱 맞게 배치
MAP_VIEW_W = MAP_VIEW_H = 0
MAP_OFFSET_X = MAP_OFFSET_Y = 0

def compute_map_viewport(world_w:int, world_h:int):
    """우측 패널/패딩을 제외한 좌측 영역에 world 비율로 맞춰 그릴 뷰포트 계산."""
    global MAP_VIEW_W, MAP_VIEW_H, MAP_OFFSET_X, MAP_OFFSET_Y
    avail_w = max(1, SCREEN_WIDTH - PANEL_RIGHT_WIDTH - 2*PADDING)
    avail_h = max(1, SCREEN_HEIGHT - 2*PADDING)
    # world 비율 유지해 최대 크기로 맞춤
    s = min(avail_w / max(1, world_w), avail_h / max(1, world_h))
    MAP_VIEW_W = max(1, int(round(world_w * s)))
    MAP_VIEW_H = max(1, int(round(world_h * s)))
    # 좌측 여백 + 세로 중앙 정렬
    MAP_OFFSET_X = PADDING + (avail_w - MAP_VIEW_W)//2
    MAP_OFFSET_Y = PADDING + (avail_h - MAP_VIEW_H)//2

def update_layout_from_env(env):
    # env.width/height가 있으면 사용, 없으면 기본 GRID 사용
    w = int(getattr(env, "width", GRID_W))
    h = int(getattr(env, "height", GRID_H))
    compute_map_viewport(w, h)

# ====== ENV / POLICY ======
def create_env_and_renderer(map_id:int, n_agents:int, world_w:int, world_h:int):
    env = FightingModel(number_agents=n_agents, width=world_w, height=world_h,
                        model_num=map_id, robot=ROBOT_VERSION_FOR_MODEL)
    renderer = make_renderer(world_w, world_h) if USE_CONTINUOUS_RENDERER else None
    update_layout_from_env(env)
    return env, renderer

def policy_fn(env: FightingModel, dt: float) -> np.ndarray:
    try:
        rx, ry = env.robot.xy
        if hasattr(env,"exit_list") and env.exit_list:
            best=(None,1e9)
            for poly in env.exit_list:
                try: cx,cy = np.mean(np.array(poly),axis=0)
                except Exception: cx,cy = poly.centroid.x, poly.centroid.y
                d=(cx-rx)**2+(cy-ry)**2
                if d<best[1]: best=((cx,cy), d)
            (cx,cy),_ = best; vx,vy = cx-rx, cy-ry
            n = math.hypot(vx,vy) or 1.0
            return np.array([1.5*vx/n, 1.5*vy/n], np.float32)
    except Exception:
        pass
    return np.array([0.0,0.0], np.float32)

def alive_agents(env)->int:
    try: return int(env.alived_agents())
    except Exception: return 0

def steps_per_second(speed:float)->float: return speed / SIM_TIMESTEP
def fmt_sim_time(step_count:int)->str:
    sec = step_count*SIM_TIMESTEP; m=int(sec//60); s=int(sec%60)
    return f"{m:02d}:{s:02d} ({sec:.1f}s)"

# =========================
# 메인
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive Crowd Simulation (Auto-fit Viewport)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18); font_small = pygame.font.SysFont("Arial", 16)

    # 상태
    paused=False; step_once=False; speed=1.0
    pending_map=6; pending_agents=20
    world_w, world_h = GRID_W, GRID_H   # ← 여기를 크게 바꿔도 화면에 맞춰서 그려짐!

    env, renderer = create_env_and_renderer(pending_map, pending_agents, world_w, world_h)
    curr_map, curr_agents = pending_map, pending_agents
    step_count=0; last=time.perf_counter(); acc_steps=0.0
    last_render_ms=0.0; target_frame_ms=1000.0/TARGET_RENDER_FPS
    substep_limit = ADAPTIVE_SUBSTEP_INIT

    # ─ GUI 배치 ─
    panel_x = SCREEN_WIDTH - PANEL_RIGHT_WIDTH + 12
    panel_w = PANEL_RIGHT_WIDTH - 24
    y=10; title_rect = pygame.Rect(panel_x,y,panel_w,24); y+=28
    BTN_H=28; GAP=6; btn_w=(panel_w - GAP*2)//3
    def toggle_pause(): nonlocal paused; paused=not paused
    btn_pause = UIButton((panel_x,y,btn_w,BTN_H), lambda:"▶ Resume" if paused else "⏸ Pause", toggle_pause, font_small)
    def do_step(): nonlocal step_once; 
    def do_step():
        nonlocal step_once
        if paused: step_once=True
    btn_step  = UIButton((panel_x+btn_w+GAP,y,btn_w,BTN_H), "Step ➜", do_step, font_small)
    def do_reset():
        nonlocal env, renderer, curr_map, curr_agents, step_count, acc_steps, paused
        env, renderer = create_env_and_renderer(pending_map, pending_agents, world_w, world_h)
        curr_map, curr_agents = pending_map, pending_agents
        step_count=0; acc_steps=0.0; paused=False
    btn_reset = UIButton((panel_x+(btn_w+GAP)*2,y,btn_w,BTN_H), "Reset ⟲", do_reset, font_small)
    y+=BTN_H+10

    speed_label_rect = pygame.Rect(panel_x,y,panel_w,18); y+=18
    slider = UISlider((panel_x,y,panel_w,28), SPEED_MIN, SPEED_MAX, speed, on_change=lambda v: None); y+=34

    def on_map_change(v): nonlocal pending_map; pending_map=int(v)
    input_map = UIInputNumber("Map ID (*)", panel_x, y, panel_w, pending_map, on_change=on_map_change, font=font_small, min_val=0); y+=52
    def on_agents_change(v): nonlocal pending_agents; pending_agents=int(v)
    step_agents = UIStepper("Agents (*)", panel_x, y, panel_w, pending_agents, 5, on_change=on_agents_change, font=font_small, min_val=0); y+=52

    widgets=[btn_pause,btn_step,btn_reset,slider,input_map,step_agents]

    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q): running=False
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_SPACE: paused=not paused
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_RIGHT:
                if paused: step_once=True
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_UP:
                speed=min(SPEED_MAX,speed*2.0); slider.value=speed
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_DOWN:
                speed=max(SPEED_MIN,speed/2.0); slider.value=speed
            for w in widgets:
                if hasattr(w,"handle"): w.handle(e)

        speed=slider.value

        # 누적을 'step' 단위로
        now=time.perf_counter(); dt=now-last; last=now
        if not paused or step_once:
            acc_steps += (dt*speed)/SIM_TIMESTEP
            acc_steps = min(acc_steps, MAX_ACCUM_STEPS)

        # 렌더 타임 기준으로 프레임당 서브스텝 한도 적응
        if last_render_ms > target_frame_ms*0.95:
            substep_limit = max(1, substep_limit-1)
        elif last_render_ms < target_frame_ms*0.60:
            substep_limit = min(MAX_SUBSTEPS_PER_FRAME_HARD, substep_limit+1)

        did_step=False; substeps=0
        while (acc_steps>=1.0) and (not paused or step_once) and (substeps<substep_limit):
            if ROBOT_CONTROL_MODE=="external":
                act=policy_fn(env, SIM_TIMESTEP)
                try: env.robot.receive_action(act.tolist())
                except Exception: env.robot.receive_action([float(act[0]), float(act[1])])
            env.step()
            step_count+=1; acc_steps-=1.0; substeps+=1; did_step=True
            if step_once: step_once=False; break

        # 자동 재시작
        alive = alive_agents(env)
        if (alive<=0) or (step_count>=MAX_STEPS):
            env, renderer = create_env_and_renderer(curr_map, curr_agents, world_w, world_h)
            step_count=0; acc_steps=0.0; paused=False
            alive = alive_agents(env)

        # ==== DRAW (항상 화면 안에 들어오도록) ====
        screen.fill(WHITE)

        # 맵 영역 (뷰포트 크기/위치가 자동으로 계산되어 있음)
        if USE_CONTINUOUS_RENDERER and renderer is not None:
            t0=time.perf_counter()
            rgb = renderer.draw(env, step=step_count)
            surf = pygame.surfarray.make_surface(np.transpose(rgb,(1,0,2)))
            if (surf.get_width(), surf.get_height())!=(MAP_VIEW_W, MAP_VIEW_H):
                surf = pygame.transform.scale(surf, (MAP_VIEW_W, MAP_VIEW_H))
            screen.blit(surf, (MAP_OFFSET_X, MAP_OFFSET_Y))
            last_render_ms = (time.perf_counter()-t0)*1000.0

        # 우측 패널
        pygame.draw.rect(screen,(248,248,248),(SCREEN_WIDTH-PANEL_RIGHT_WIDTH,0,PANEL_RIGHT_WIDTH,SCREEN_HEIGHT))
        pygame.draw.line(screen,GREY,(SCREEN_WIDTH-PANEL_RIGHT_WIDTH,0),(SCREEN_WIDTH-PANEL_RIGHT_WIDTH,SCREEN_HEIGHT),2)
        title = pygame.font.SysFont("Arial",20).render("Simulation Control",True,DARK)
        screen.blit(title,(title_rect.left,title_rect.top))

        # 위젯
        screen.blit(font_small.render("Speed",True,DARK),(speed_label_rect.left,speed_label_rect.top))
        for w in widgets:
            if hasattr(w,"draw"): w.draw(screen)

        # 설명(Status) — 패널 하단 고정
        info=[
            f"Paused:{paused}  Speed:{speed:.2f}x  SPS:{steps_per_second(speed):.1f}  SubstepLimit:{substep_limit}",
            f"Step:{step_count}  Sim:{fmt_sim_time(step_count)}  Alive:{alive}",
            f"World:{int(getattr(env,'width',world_w))}x{int(getattr(env,'height',world_h))}  Viewport:{MAP_VIEW_W}x{MAP_VIEW_H}",
            f"LastRender:{last_render_ms:.1f} ms  Target:{(1000/TARGET_RENDER_FPS):.1f} ms   (Reset to apply Pending)",
            f"Pending(*) Map:{pending_map}  Agents:{pending_agents}    Keys: Space/→/↑/↓/ESC",
        ]
        lh=18; total_h=len(info)*lh+8; info_y=SCREEN_HEIGHT-total_h
        for line in info:
            screen.blit(font_small.render(line,True,(60,60,60)),(panel_x,info_y)); info_y+=lh

        pygame.display.flip()
        clock.tick(TARGET_RENDER_FPS)

    pygame.quit()

if __name__=="__main__":
    main()
