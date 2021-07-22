'''
이 버전은 deflection angle에 의해 컨트롤하는 case 2 입니다.

저희의 목표는 target까지 최대한 빠르게 이동해 target을 맞추면 됩니다.

이 예제는 카트폴 예제를 참고하였습니다.

1. Fortress Invasion System

    (1) Requirements : 

        state = (x, y, v, psi)
        
        목표 : fortress 파괴, bullet, wall, Obstacle과 충돌하면 게임 종료


    (2) case 1 : 측면 가스분사

        condition : 진공, 좌우 측면에 가스를 분사하여 모멘트를 발생시키는 상황 가정

        Eqn. of motion :

            (x_dot, y_dot, V_dot, psi_dot) = (V_cospsi, V_sinpsi, 0, F/Vm), (Input : F)
            

    (3) case 2 : Deflection에 의한 Torque

        condition : C_nr = 0, none sideslip

        Eqn. of motion : 

            psi_ddot = +-0.3

            (x_dot, y_dot, V_dot, psi_dot) = (V_cospsi, V_sinpsi, 0, psi_ddot * dt), (Input : Psi_ddot)




2. Fortress Invasion environment
    
    (1) Rewad 기준 : 

        1. 직진(1), 좌우 회전(1.2)
        
        2. 성공(100)
        
        3. 40넘게 움직이면서 wall에 충돌(-20) : 
            
            성공할 경우에도 마찬가지, 보통 30 이하의 움직임으로도 충분히 성공가능하다. 더 빠르게 성공시키기 위해 설정
        
        4. 50넘게 움직일 경우(-40)
            
            허용하는 움직임의 개수가 많아질 수록 화면 내부에서 뱅글뱅글 도는 경우가 생긴다. 
            
            이를 방지하기 위해 허용할 수 있는 움직임의 최대값 설정

    (2) 코드 구성

        1. __init__ : 

            초기설정, state, 상,하한선, seed, viewer, step beyond done

        2. seed : sedd 데이터 반환

        3. step : state 업데이트, step byond done 여부 파악, reward 판정, 충돌감지(wall, fortress)

        4. reset

        5. render

        6. close





'''
import pyglet
import gym
from gym import spaces, logger
# from gym.envs.classic_control.rendering import LineWidth, Viewer, get_window
from gym.utils import seeding
import cv2
import math
import time 
import os, sys
from datetime import datetime
import numpy as np
# import pygame
# from pyglet.text import Label



class obj:
    def create(self,x,y,sx,sy):
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy

fortress = obj()
fortress.create(8,8,2,2)

obstacle1 = obj()
obstacle1.create(2,6,2,2)

def cal_dis(self,x,y):
        self.distance = math.sqrt(pow(fortress.x+fortress.sx-x, 2)+pow(fortress.y+fortress.sy-y,2))
        # print("distance ratio = {}".format(1-self.distance))
        distance_long = math.sqrt(pow(fortress.x+fortress.sx, 2)+pow(fortress.y+fortress.sy,2))
        distance_ratio = 1-self.distance/distance_long #가까이 올수록 감소
        return distance_ratio
        

# class DrawText:
#     def __init__(self, label:pyglet.text.Label):
#         self.label=label
#     def render(self):
        
#         self.label.draw()
#     def delete(self):
#         self.label.delete()





class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    # 1. init function : 초기 parameter 설정

    def __init__(self):

        #state 설정
        self.theta = 0
        self.theta_dot = 0       # 회전 각속도
        self.theta_ddot = 0
        self.x = 0
        self.y = 0
        self.velocity = 0.7
        
        # state의 한계점 설정
        self.y_threshold = 10 #한계 지정
        self.x_threshold = 10

        high = np.array([self.x_threshold ,
                         np.finfo(np.float32).max,
                         self.y_threshold , 
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        
        # Example for using image as input:

        self.observation_space = spaces.Box(-high, high,dtype=np.float32)

        self.seed()
        self.viewer = None
        self.parameter = None
        self.state = None

        self.steps_beyond_done = None



    # 2. seed : 초기 시드값 설정

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




    # 3. step function
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        
        # state 설정

        x, y, v, psi = self.state
        if action == 2:
            self.theta_ddot = 0
            
        elif action==0:
            self.theta_ddot = 0.7
            
        else:
            self.theta_ddot = -0.7#action 1 = left
        self.theta_dot += self.theta_ddot/self.velocity
        self.theta += self.theta_dot
        
        v = self.velocity
        psi = self.theta%360
        x -= v*math.sin(math.radians(psi))
        y += v*math.cos(math.radians(psi))
        
        
        d = math.sqrt(pow(fortress.x+fortress.sx-x, 2)+pow(fortress.y+fortress.sy-y,2))
        phi = math.degrees(math.atan(y/x)) - psi

        # state 갱신
        self.state = (x, y, v, psi)


        # 프레임에서 벗어나는지 아닌지 판단

        done = bool( 
        x < -10
        or x > 10
        or y > 10
        or y < -1
        ) #하면 안되는 짓 = True 반환, 안벗어나면 False
        
        fortress_hit = 0
        obstacle1_hit = 0
        
        if (x>= fortress.x) and (fortress.x+fortress.sx >= x):
            if (y >= fortress.y) and (fortress.y+fortress.sy >= y):
                done = True
                fortress_hit = 1
            
        if (x>= obstacle1.x) and (obstacle1.x+obstacle1.sx >= x):
            if (y >= obstacle1.y) and (obstacle1.y+obstacle1.sy >= y):
                done = True
                obstacle1_hit = 1

                
                


        if not done: # not True = False,안벗어난다면,
            
            
            if d<100 :
                reward = 8
                if math.cos(phi)>0.866:
                    rewrad = 10
                
            
                
            else : 
                reward = -1+math.cos(phi)

        elif self.steps_beyond_done is None: 
            # Pole just fell!
            self.steps_beyond_done = 0
            
            if fortress_hit == 1:
                reward = 120
                print('Yes!!')
                
            if obstacle1_hit == 1:
                reward = -120
                print('No!!!')
            
            else:
                distance_ratio = cal_dis(self, x, y)
                reward = 120*(distance_ratio)
#                 reward = 5
                print("distance_ratio = {}".format(distance_ratio))
#                 print("reward = {}".format(reward))
        else:
            if self.steps_beyond_done == 0: #이 에피소드는 끝남.
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward =0


        return np.array(self.state), reward, done, {}







    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None #초기화
        self.theta = 0
        self.theta_dot = 0 
        return np.array(self.state)








    def render(self, mode='human'):

        

        screen_width = 800
        screen_height = 800

        world_width = self.x_threshold
        world_height = self.y_threshold
        scale_x = screen_width/world_width
        scale_y = screen_height/world_height
        x = self.state
        cartx = x[0] * scale_x  # MIDDLE OF CART x의 진짜 위치
        carty = x[1] * scale_y
        psi = x[3]



        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(800,800)
            # self.viewer = pyglet.canvas.get_display(screen_width, screen_height)
            self.viewer.set_bounds(-100,800,-100,800)
            # axleoffset = cartheight / 4.0 #오프셋?
            
            
 
            
            sky = rendering.FilledPolygon([(-100, -100), (-100, 800), (800, 800), (800, -100)])
           
            self.skytrans = rendering.Transform() #트랜스폼 여기서는 0
            sky.add_attr(self.skytrans) #카트 자세 지정
            self.viewer.add_geom(sky) #최종 지정
            sky.set_color(0.15, 0.145, 0.42)







            cart = rendering.make_polyline([(-7.5,-5), (-7.5, 5), 
                                             (0,12.5),(7.5,5), (7.5,-5), 
                                             (2.5,-5), (7.5,-10), (2.5,-10), 
                                             (0,-15), (-2.5,-10),(-7.5,-10),
                                             (-2.5, -5),(-7.5,-5)]) #좌표로 채워진 다각형 생성
            self.carttrans = rendering.Transform() #트랜스폼 여기서는 0
            cart.add_attr(self.carttrans) #카트 자세 지정
            cart.set_color(255, 255, 255)
            self.viewer.add_geom(cart) #최종 지정







            fortress = rendering.FilledPolygon([(0, 0), (0, 160), (160, 160), 
                                                (160, 0)])            
            self.forttrans = rendering.Transform() #트랜스폼 여기서는 0
            fortress.add_attr(self.forttrans) #카트 자세 지정
            self.viewer.add_geom(fortress) #최종 지정
            fortress.set_color(1, .27, 0)


            obstacle1 = rendering.FilledPolygon([(0, 0), (0, 160), (160, 160), 
                                                (160, 0)])            
            self.obstacle1 = rendering.Transform() #트랜스폼 여기서는 0
            fortress.add_attr(self.forttrans) #카트 자세 지정
            self.viewer.add_geom(fortress) #최종 지정
            fortress.set_color(1, .27, 0)

            

            
            # self.parameter = pyglet.canvas.get_display()

            

        
            
  

        if self.state is None:
            return None


        
        self.carttrans.set_translation(cartx, carty)
        self.carttrans.set_rotation(math.radians(psi))


        fortx = 640
        forty = 640
        self.forttrans.set_translation(fortx, forty)
#         text = 'psi = {}'.format(round(360-psi,1))
#         text2 = 'psi_dot ={}'.format(round(self.theta_dot, 1))
#         text3 = 'psi_ddot = {}'.format(self.theta_ddot)
        
        
#         self.Label = pyglet.text.Label(text, font_size=36,
#                         x=10, y=200, anchor_x='left', anchor_y='bottom',
#                         color=(255, 255, 0, 50))
#         self.Label2 = pyglet.text.Label(text2, font_size=36,
#                         x=10, y=140, anchor_x='left', anchor_y='bottom',
#                         color=(255, 255, 0, 50))
#         self.Label3 = pyglet.text.Label(text3, font_size=36,
#                         x=10, y=80, anchor_x='left', anchor_y='bottom',
#                         color=(255, 255, 0, 50))


#         self.Label.draw()
#         self.viewer.add_onetime(DrawText(self.Label))
#         self.Label2.draw()
#         self.viewer.add_onetime(DrawText(self.Label2))
#         self.Label3.draw()
#         self.viewer.add_onetime(DrawText(self.Label3))
       

#         return self.viewer.render(return_rgb_array=False)




    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    



    def path_save(self,count,done,episode):
        if episode == 0 and count ==0:
            path_saved = open("path.txt","w",encoding = "utf8")
            print("new Training!!", file = path_saved)
            print("first step success")

        path_saved = open("path.txt","a",encoding = "utf8")
        
        if count == 0:
            print("new episode", file = path_saved)
            print("second step success")

        print(self.state, file = path_saved)

        if done == True:
            path_saved.close()
    