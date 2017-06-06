import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

actions_space=[Action.LEFT,Action.RIGHT,Action.ACCELERATE,Action.BREAK]
n_actions= len(actions_space)

class QAgent(Agent):
    def __init__(self, learning_rate=0.01, epsilon=0.9, reward_decay=0.9):
        super(QAgent, self).__init__()
        # Add member variables to your class here
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon= epsilon
        self.q_table = pd.DataFrame(columns=actions_space)
        self.total_reward_list=[]
        self.reward=0
        self.total_epoch=500

    def initialise(self, grid):
        self.reward=0
        self.total_reward = 0
        self.s="4xxxxxxxxxx"
        self.s_=""
    def act(self):
        # #state_action = self.q_table.iloc[self.cur_state,:]
        self.check_ifstate_exist(self.s)
        action=Action.ACCELERATE
        if np.random.uniform() < self.epsilon:
            state_action= self.q_table.ix[self.s,:]
            state_action= state_action.reindex(np.random.permutation(state_action.index))
            action=state_action.argmax()
        else:
            flag=np.random.randint(1,5)
            if flag== 1:
                action=Action.LEFT
            if flag== 2:
                action=Action.RIGHT
            if flag== 3:
                action=Action.ACCELERATE
            if flag==4:
                action=Action.BREAK
        self.reward=self.move(action)
        self.total_reward +=self.reward
        return action

    def sense(self,grid):
        self.s_="xxxxxxxxxx"
        i=0
        for i in range (1,9):
            if grid[0][i]==2:
                i=i
                break
        self.s_=str(i)+self.s_
        for j in range(0,10):
            for k in range(0,8):
                if grid[k][j]==1:
                    self.s_ = self.s_[:j]+str(k)+self.s_[j+1:]
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self,action):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        self.check_ifstate_exist(self.s_)
        q_predict= self.q_table.ix[self.s,action]
        q_target= self.reward+self.gamma*self.q_table.ix[self.s_,:].max()
        self.q_table.ix[self.s,action] +=self.lr*(q_target-q_predict)
        self.s=self.s_
    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        if iteration >= 6500:
            self.total_reward_list.append(self.total_reward)
        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(1)

    def check_ifstate_exist(self,state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(actions_space),
                    index=self.q_table.columns,
                    name=state,
                )

            )

if __name__ == "__main__":
    a = QAgent()
    a.run_Q(True, episodes=a.total_epoch, draw=True)
    print 'Total reward: ' + str(a.total_reward)
    print a.total_reward_list
    print("mean is:{0}".format(np.mean(a.total_reward_list)))
    print("variance is:{0}.".format(np.var(a.total_reward_list)))
    #plot total rewards and distributation
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)

    p1.plot(range(1,a.total_epoch+1), a.total_reward_list)
    p1.set_xlabel("episode")
    p1.set_ylabel("total reward")

    p2.hist(a.total_reward_list)
    p2.set_xlabel("total reward")
    p2.set_ylabel("number")

    plt.show()