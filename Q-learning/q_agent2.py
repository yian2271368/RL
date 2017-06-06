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
    def __init__(self, learning_rate=0.01, epsilon=0.9, reward_decay=0.8):
        super(QAgent, self).__init__()
        # Add member variables to your class here
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon= epsilon
        self.q_table = pd.DataFrame(columns=actions_space)
        self.total_reward_list=[]
        #self.cur_state= np.nan
        #self.next_state=
        self.reward=0
        # self.var_list=[]
        # self.mean_list=[]
        self.total_epoch=500
        # self.reward_list=[]
        # self.reward_total_list=[]


    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        # self.cur_state=unicode(grid)
        # self.check_ifstate_exist(self.cur_state)
        self.reward=0
        self.total_reward = 0
        # self.q_table.info()
        # self.q_table.head(10)
        # self.cur_state=grid.flatten()
        # self.cur_state=np.array2string(self.cur_state)
        #self.cur_state=str(grid[0:2,:])
        #self.check_ifstate_exist(str(self.cur_state))

    # def everyepoch(self):
    #     p=sum(self.reward_list)
    #     mean=p/len(self.reward_list)*1.0
    #     variance=np.var(self.reward_list)
    #     self.var_list.append(variance)
    #     self.mean_list.append(mean)
    #     self.reward_total_list.append(p)
    #     del self.reward_list[:]
    #
    # def create_image(self):
    #     x_axis=[]
    #     figurename=["variance","mean","reward","reward distribution"]
    #     datasource=[self.var_list,self.mean_list,self.reward_total_list]
    #     for i in range(self.total_epoch):
    #         x_axis.append(i+1)
    #     for i in range (len(figurename)-1):
    #         plt.plot(x_axis, datasource[i]);
    #         plt.xlabel("Epoch Number")
    #         plt.ylabel(figurename[i])
    #         plt.grid(True)
    #         plt.show()
    #         plt.savefig(figurename[i])
    #
    #     plt.scatter(x_axis, datasource[2]);
    #     plt.xlabel("Epoch Number")
    #     plt.ylabel(figurename[3])
    #     plt.grid(True)
    #     plt.show()
    #     plt.savefig(figurename[3])

    def act(self,ob):
        # #state_action = self.q_table.iloc[self.cur_state,:]
        self.check_ifstate_exist(ob)
        action=Action.ACCELERATE
        if np.random.uniform() < self.epsilon:
            state_action= self.q_table.ix[ob,:]
            state_action= state_action.reindex(np.random.permutation(state_action.index))
            action=state_action.argmax()
            #print ("+++++++++:{0}".format(action))
            # if action_name==1:
            #     action=Action.ACCELERATE
            # if action_name==5:
            #     action=Action.BREAK
            # if action_name==11:
            #     action=Action.RIGHT
            # if action_name==12:
            #     action=Action.LEFT

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
        #self.reward += self.move(action)
        #self.total_reward += self.reward
        self.total_reward +=self.reward
        #self.reward_list.append(reward)
        return action

    def sense(self,grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        # m=np.where(grid==2)
        # n1=m[0]
        # n2=m[1]
        # p1=n1[0]
        # p2=n2[0]
        # index_agent=np.where(grid==2)
        # index_cars=np.where(grid[:8,:]==1)
        #self.next_state=unicode(grid)
        # self.next_state=grid.flatten()
        # self.next_state=np.array2string(self.next_state)
        #self.next_state=str(grid[0:2,:])
        #self.check_ifstate_exist(ob_)
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self,action,ob,ob_):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        self.check_ifstate_exist(ob_)
        q_predict= self.q_table.ix[ob,action]
        #no need to worry about terminal
        # print 'check point ++++++++++++++++++++++++++++++++++++++++++++'
        # print self.total_reward
        # print(self.cur_state)
        # print self.next_state
        # print self.q_table.ix[self.next_state,:].max()
        #self.q_table.info()
        q_target= self.reward+self.gamma*self.q_table.ix[ob_,:].max()
        self.q_table.ix[ob,action] +=self.lr*(q_target-q_predict)

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        if iteration >= 6500:
            self.total_reward_list.append(self.total_reward)
        # Show the game frame only if not learning
        #if not learn:
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
    a.run_Q_new(True, episodes=a.total_epoch, draw=True)
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