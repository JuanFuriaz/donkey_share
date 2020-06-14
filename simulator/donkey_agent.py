# environment wrapper for donkey car
# in simulator we use DonkeyGymEnv from dgym; 
# in real world we have the camera and the ps3-controller

from threading import Thread

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
#from donkey_gym.envs.donkey_sim import DonkeyUnitySim
import time
from config import MIN_STEERING, MAX_STEERING, JERK_REWARD_WEIGHT, MAX_STEERING_DIFF, MIN_THROTTLE, MAX_THROTTLE, NEGATIVE_REWARD_WEIGHT, POSITIVE_REWARD_WEIGHT, N_COMMAND_HISTORY
import random
import tensorflow as tf
import matplotlib
import cv2
import math
import PIL
import io
from torchvision.transforms import transforms
import torch

class DonkeyAgent(gym.Env):

    # control actions for donkey-car
    ACTION = ["steering", "throttle"]
    
    # environment; in simulator we use class DonkeyGymEnv from dgym; 
    # in real world we have PiCamera from camera.py
    wrapped_env = None
    env_type = None
    controller = None
    
    # save lifetime on track
    lifetime = 0
    
    # command history
    command_history = []
    
    last_user_action = None
    
    # greyscale to reduce complexity
    use_greyscale = False
    
    # crop augmentation to reduce complexity
    use_crop_augment = False
    top_percent= 50
	
    vae = None
    device = None
    best_lifetime = 0

    

    def __init__(self, _wrapped_env, time_step=0.05, frame_skip=2,env_type='simulator', controller=None, vae = None, device = None):
        
        # min- max
        self.set_action_space(MIN_STEERING, MAX_STEERING, MIN_THROTTLE, MAX_THROTTLE)
        #self.set_action_space(0, MIN_STEERING+MAX_STEERING, MIN_THROTTLE, MAX_THROTTLE)
        
        self.vae = vae
        self.device = device

        # Camera sensor data
        self.observation_space = spaces.Box(0, 255, self.viewer_get_sensor_size())

        # Frame Skipping
        self.frame_skip = frame_skip

        # Simulation related variables.
        self.seed()
        
        self.wrapped_env = _wrapped_env
        self.env_type = env_type
        self.controller = controller
		
        if vae:
            self.use_crop_augment = False
            self.use_greyscale = False
			
        self.n_commands = 2
        self.n_command_history = N_COMMAND_HISTORY
			
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        
    def viewer_get_sensor_size(self):
        height = 120
        width = 160
        dim = 3
      
        if self.use_greyscale:
            dim = 1
        
        if self.use_crop_augment:
            top = math.ceil(self.top_percent*120/100)
            height = height -top
        
        camera_img_size=(height, width, dim)
        
        if self.vae:
            #return (232,)
            return (52,)
        
        return camera_img_size
  
  
    def jerk_penalty(self):
        jerk_penalty = 0
        if len(self.command_history)>2:
            last_command = self.command_history[-1]
            prev_last_command = self.command_history[-2]
            steering = last_command['angle']
            prev_steering = prev_last_command['angle']
            steering_diff = (prev_steering - steering) / (MAX_STEERING - MIN_STEERING)
            
            if abs(steering_diff) > MAX_STEERING_DIFF:
                error = abs(steering_diff) - MAX_STEERING_DIFF
                jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
            else:
                jerk_penalty += 0
                
        return jerk_penalty
            
    
    def calc_reward(self,action):
        done = self.is_game_over()

        throttle = action[1]
        reward = 0
        angle =  action[0]
        
        
        if done:
            # something went wrong; car got off the track
            reward = -10 - NEGATIVE_REWARD_WEIGHT * throttle
            
            if self.lifetime < self.best_lifetime:
                reward = reward - (self.best_lifetime-self.lifetime)*0.1
            
            self.lifetime = 0
        else:
            # car is on the track
            self.lifetime = self.lifetime + 1
            
            if self.lifetime > self.best_lifetime:
                self.best_lifetime = self.lifetime
                
            reward = 1 + 1 * (throttle)
            
            jerk_penalty = self.jerk_penalty()
            reward = reward - jerk_penalty


        return reward
    
    def set_action_space(self,min_steer,max_steer,min_throttle,max_throttle):
        self.action_space = spaces.Box(np.array([min_steer,max_steer]),np.array([min_throttle,max_throttle]))
        
        
    def crop_image(self, img_arr):
        height, width, _ = img_arr.shape
        self.bottom_percent = 0
        self.left_percent = 0
        self.right_percent = 0
        
        top = math.ceil(self.top_percent*height/100)
        bottom = math.ceil(self.bottom_percent*height/100)
        left = math.ceil(self.left_percent*width/100)
        right = math.ceil(self.right_percent*width/100)
        img_arr = img_arr[top:height-bottom, 
                          left: width-right]
        
        return img_arr
		
		
    def _record_action(self, action):

        if len(self.action_history) >= self.n_command_history * self.n_commands:
            del self.action_history[:2]
        for v in action:
            self.action_history.append(v)
		
    def _encode_image(self, image):
        observe = PIL.Image.fromarray(image)
        observe = observe.resize((160,120))
        croped = observe.crop((0, 40, 160, 120))
        #self.teleop.set_current_image(croped)
        tensor = transforms.ToTensor()(croped)
        tensor.to(self.device)
        z, _, _ = self.vae.encode(torch.stack((tensor,tensor),dim=0)[:-1].to(self.device))
        return z.detach().cpu().numpy()[0]

    def _postprocess_observe(self,observe, action):
        self._record_action(action)
        print('donkey_agent: vae encoding image...')
        observe = self._encode_image(observe)
        #print('OBS: %s'%str(observe))
        if True and self.n_command_history > 0:
            observe = np.concatenate([observe, np.asarray(self.action_history)], 0)
            

        return observe
    
    def viewer_observe(self, action):
        if self.env_type == 'simulator':
            observation = self.wrapped_env.frame
            # calc reward
            reward = self.calc_reward(action)
            done = (self.is_game_over())
            info = self.wrapped_env.info
            
            if self.use_crop_augment:
                observation = self.crop_image(observation)
            
            if self.use_greyscale:
                observation = tf.image.rgb_to_grayscale(observation)
                observation = observation.eval(session=tf.compat.v1.Session())
				
            if self.vae:
                observation = self._postprocess_observe(observation, action)
            
            return  observation, reward, done, info
        else:
            observation = self.wrapped_env.frame
            reward = 1
            done = self.is_game_over()
            info = self.wrapped_env.info
            return  observation, reward, done, info
        
	
	
    
    def viewer_take_action(self,action):
        if self.env_type == 'simulator':
            # environment is the simulator
            #self.wrapped_env.action = action
            if self.controller.mode == 'user':
                self.command_history = []
                steering = self.controller.angle
                throttle = self.controller.throttle
                self.last_user_action = [steering,throttle]
                print('self.last_user_action: ' + str(self.last_user_action ))
                return

                
            self.controller.throttle = action[1]
            self.controller.angle = action[0]
            print('angle:' + str(self.controller.angle))
            print('throttle:' + str(self.controller.throttle))
            
            # save command
            command = {'angle':action[0], 'throttle':action[1]}
            self.command_history.append(command)
            self.last_user_action = None
            #time.sleep(0.1)
            
        else:
            # environment is the real world
            self.controller.throttle = action[0]
            self.controller.angle = action[1]
            

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        # Clip steering angle rate to enforce continuity
        if True and len(self.command_history) > 0:
                #print("COMMAND_HIST:%s"%str(self.command_history))
                prev_steering = (self.command_history[-1])['angle']
                print('PREV_STEERING:%s'%str(prev_steering))
                max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
                diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
                action[0] = prev_steering + diff                

        
        for i in range(self.frame_skip):
            self.viewer_take_action(action)
            observation, reward, done, info = self.viewer_observe(action)
        return observation, reward, done, info

    def reset(self):
        print('RESET')
        self.viewer_take_action([0,0])
        self.lifetime = 0
        command_history = []
        self.action_history = [0.] * (self.n_command_history * self.n_commands)
        observation, reward, done, info = self.viewer_observe([0,0])
        return observation

    def render(self, mode="human", close=False):
        '''if close:
            self.viewer.quit()

        return self.viewer.render(mode)'''
        return self.wrapped_env.frame

    def is_game_over(self):
        if self.controller.mode == 'user':
            return True
        else:
            return False

    def get_last_user_action(self):
        # no action taken
        if self.last_user_action == [0,0]:
            return None
        return self.last_user_action
    

        
