import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
import gym
import numpy as np 
import random as rand
from FI_env import CustomEnv
import FI_env





LOSS_CLIPPING = 0.2 # 상한과 하한
class Agent(object):
    def __init__(self):
        self.env = CustomEnv()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.value_size = 1
        
        self.node_num = 24
        self.learning_rate_actor = 0.0005 #학습 속도 정책
        self.learning_rate_critic = 0.0001
        self.epochs_cnt = 10 #수집된 데이터를 몇번 반복해서 학습할지 정하는 파라미터
        self.model_actor = self.build_model_actor()# 모델 생성
        self.model_critic = self.build_model_critic()
        
        self.discount_rate = 0.99
        self.smooth_rate = 0.95 #GAE계산할 때 사용하는 Tunning 파라미터
        self.penalty = 1
        #---------------------------------------------------
        self.episode_num = 500
        self.mini_batch_step_size = 32        
        
        self.moving_avg_size = 20
        self.reward_list= []
        self.count_list = []
        self.moving_avg_list = []
        
        self.states, self.states_next, self.action_matrixs = [],[],[]
        self.dones, self.action_probs, self.rewards = [],[],[]
        self.DUMMY_ACTION_MATRIX = np.zeros((1,1,self.action_size))
        self.DUMMY_ADVANTAGE = np.zeros((1,1,self.value_size))
        
        
    class MyModel(tf.keras.Model):
        def train_step(self, data):

            in_datas, out_action_probs = data #입력변수 설정
            states, action_matrixs, advantages = in_datas[0], in_datas[1], in_datas[2] #입력변수 설정/ 에피소드의 count만큼의 배열이 저장되어있음
            

            with tf.GradientTape() as tape:
                y_pred = self(states, training=True) #새로운 정책(Update된 것은 아님), states에서 예를 들어 16번의 타임 스텝을 거치면(Count) y_pred또한 16개의 새로운 정책
                new_policy = K.max(action_matrixs*y_pred, axis=-1)   #행동 중 어떤 행동(예전 정책)을 선택하냐에 따라 일의 위치가 달라짐
                old_policy = K.max(action_matrixs*out_action_probs, axis=-1)   # *액션의 확률
                r = new_policy/(old_policy) #옛 정책과 신규 정책의 비 / 중요도 샘플링
                clipped = K.clip(r, 1-LOSS_CLIPPING, 1+LOSS_CLIPPING)#하한과 상한
                loss = -K.minimum(r*advantages, clipped*advantages) #Loss function, 클리핑
            #---------------------------------------------
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
    def build_model_actor(self):
        input_states = Input(shape=(1,self.state_size), name='input_states')
        input_action_matrixs = Input(shape=(1,self.action_size), name='input_action_matrixs')
        input_advantages = Input(shape=(1,self.value_size), name='input_advantages')

        x = (input_states)
        x = Dense(self.node_num, activation='relu')(x)
        out_actions = Dense(self.action_size, activation='softmax', name='output')(x)
        
        model = self.MyModel(inputs=[input_states, input_action_matrixs, input_advantages], outputs=out_actions)
        model.compile(optimizer=Adam(lr=self.learning_rate_actor))
        
        model.summary()
        return model
    
    def build_model_critic(self):
        input_states = Input(shape=(1,self.state_size), name='input_states')
        x = (input_states)
        x = Dense(self.node_num, activation='relu')(x)
        out_values = Dense(self.value_size, activation='linear', name='output')(x)
        
        model = tf.keras.models.Model(inputs=[input_states], outputs=[out_values])
        model.compile(optimizer=Adam(lr=self.learning_rate_critic),
                      loss='mean_squared_error'
                     )
        model.summary()
        return model

    def train(self):
        for episode in range(self.episode_num):
            

            state = self.env.reset()
            self.env.max_episode_steps = 500 #최대 실행 횟수
            

            count, reward_tot = self.make_memory(episode, state)#경험 수집
            self.train_mini_batch()#모델 학습
            self.clear_memory() #정리
            
            # if count < 500: #일찍 끝나면
            #     reward_tot = reward_tot-self.penalty
                
            self.reward_list.append(reward_tot)
            self.count_list.append(count)
            self.moving_avg_list.append(self.moving_avg(self.count_list,self.moving_avg_size))                
            
            if(episode % 10 == 0):
                print("episode:{}, moving_avg:{}, rewards_avg:{}".format(episode, self.moving_avg_list[-1], np.mean(self.reward_list)))

        self.save_model()
        
    def make_memory(self, episode, state):
        reward_tot = 0
        count = 0
        reward = np.zeros(self.value_size)
        advantage = np.zeros(self.value_size)
        target = np.zeros(self.value_size)
        action_matrix = np.zeros(self.action_size)
        done = False

        

        while not done: # 에피소드 종료 조건
            count+=1
            

            state_t = np.reshape(state,[1, 1, self.state_size])
            action_matrix_t = np.reshape(action_matrix,[1, 1, self.action_size])
            
            action_prob = self.model_actor.predict([state_t, self.DUMMY_ACTION_MATRIX, self.DUMMY_ADVANTAGE]) #행동 예측

            action = np.random.choice(self.action_size, 1, p=action_prob[0][0])[0] # 행동 선택
            action_matrix = np.zeros(self.action_size) #초기화
            action_matrix[action] = 1

            state_next, reward, done, none = self.env.step(action)
            self.env.render() #####
            
            if episode % 10 ==0 : #or if fortress_hit == 1             
                 self.env.path_save(count,done,episode)
            
            state_next_t = np.reshape(state_next,[1, 1, self.state_size])
            
            # if count < 500 and done:
            #     reward = self.penalty 
        
            self.states.append(np.reshape(state_t, [1,self.state_size]))
            self.states_next.append(np.reshape(state_next_t, [1,self.state_size]))
            self.action_matrixs.append(np.reshape(action_matrix, [1,self.action_size]))
            self.dones.append(np.reshape(0 if done else 1, [1,self.value_size])) #진행되면 0, 종료되면 1
            self.action_probs.append(np.reshape(action_prob, [1,self.action_size]))
            self.rewards.append(np.reshape(reward, [1,self.value_size]))
            

                
            
            

            if(count % self.mini_batch_step_size == 0):
                self.train_mini_batch()
                self.clear_memory()


            reward_tot += reward
            state = state_next

            if count > 150:
                done = True
                print('Force quit')
                reward_tot = 0
#                 print("reward = {}".format(reward))
            

            
        
        print('total_reward = {0}'.format(reward_tot))
        return count, reward_tot
        

        

    def make_gae(self, values, values_next, rewards, dones): #할인율 + 어드벤티지or 타겟
        delta_adv, delta_tar, adv, target = 0, 0, 0, 0
        advantages = np.zeros(np.array(values).shape)
        targets = np.zeros(np.array(values).shape)
        for t in reversed(range(0, len(rewards))):
            delta_adv = rewards[t] + self.discount_rate * values_next[t] * dones[t] - values[t] #스텝별 어드밴티지 구하는 공식
            delta_tar = rewards[t] + self.discount_rate * values_next[t] * dones[t] #스텝별 타겟 계산
            adv = delta_adv + self.smooth_rate*self.discount_rate * dones[t] * adv #할인된 어드밴티지 계싼
            target = delta_tar + self.smooth_rate*self.discount_rate * dones[t] * target #할인된 타겟 계산/ Smooth_rate PPO 튜닝의 장치
            advantages[t] = adv
            targets[t] = target
        return advantages, targets

    def train_mini_batch(self):
        
        if len(self.states) == 0:
            return
        
        states_t = np.array(self.states)
        states_next_t = np.array(self.states_next)
        action_matrixs_t = np.array(self.action_matrixs)
        action_probs_t = np.array(self.action_probs)
        rewards_t = np.array(self.rewards)
        #----클래스 변수 저장값 불러오기, 실행된 count만큼의 사이즈를 가지고 있음

        values = self.model_critic.predict(states_t)
        values_next = self.model_critic.predict(states_next_t) #가치 예측
        
        advantages, targets = self.make_gae(values, values_next, self.rewards, self.dones)#GAE 계산
        advantages_t = np.array(advantages)
        targets_t = np.array(targets)
        self.model_actor.fit([states_t, action_matrixs_t, advantages_t], [action_probs_t], epochs=self.epochs_cnt, verbose=0) #모델 학습
        self.model_critic.fit(states_t, targets_t, epochs=self.epochs_cnt, verbose=0)       
 
    def clear_memory(self):
        self.states, self.states_next, self.action_matrixs = [],[],[]
        self.dones, self.action_probs, self.rewards = [],[],[]
        
    def moving_avg(self, data, size=10):
        if len(data) > size:
            c = np.array(data[len(data)-size:len(data)]) 
        else:
            c = np.array(data) 
        return np.mean(c)
    
    def save_model(self):
        self.model_actor.save("./model/ppo")
        print("*****end learing")

if __name__ == "__main__":
    agent = Agent()
    agent.train()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(agent.reward_list, label='rewards')
plt.plot(agent.moving_avg_list, linewidth=4, label='moving average')
plt.legend(loc='upper left')
plt.title('PPO')
plt.show()
