import cv2
import numpy as np

from enduro.agent import Agent
from enduro.action import Action
import matplotlib.pyplot as plt

num_features= 10
actions=[Action.ACCELERATE,Action.BRAKE,Action.RIGHT,Action.LEFT]
class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.lr = 0.01
        self.total_reward_list = []
        self.gamma = 0.9
        self.reward = 0
        self.epsilon= 0.9
        self.weights= np.random.rand(num_features)
        self.action= Action.ACCELERATE
        self.episodes= 2000
        self.weights_list=[]
        self.state=[]
    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        self.next_state = (road,cars,speed,self.buildstate(grid))
        self.total_reward_list.append(self.total_reward)
        # You could comment this out in order to speed up iterations
        self.weights_list.append(self.weights)
        self.total_reward = 0

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        self.state = self.next_state
        if np.random.uniform()>self.epsilon:
            flag=np.random.randint(1,5)
            if flag==1:
                self.action=Action.LEFT
            if flag==2:
                self.action=Action.RIGHT
            if flag==3:
                self.action=Action.ACCELERATE
            if flag==4:
                self.action=Action.BRAKE
        self.reward= self.move(self.action)
        self.total_reward += self.reward

        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        self.next_state=(road,cars,speed,self.buildstate(grid))
    def learn(self,grid):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        self.features= self.buildfeatures(self.next_state,self.action,grid)
        self.q_target= np.dot(self.weights,self.features)
        q_predict_list=[]
        features_list=[]
        for action in actions:
            features_list.append(self.buildfeatures(self.next_state,action,grid))
        for i in range(len(features_list)):
            q_predict_list.append(np.dot(self.weights,features_list[i]))
        max_q_predict_index= np.argmax(q_predict_list)   #choose the max q value
        self.features = features_list[max_q_predict_index]  #update features
        self.action=actions[max_q_predict_index]
        self.q_predict= q_predict_list[max_q_predict_index]

        error=self.reward+self.gamma*self.q_predict-self.q_target
        self.weights =self.weights+ error*self.lr*self.features
        #print self.features
        #print '++++++++++'
        #print self.weights

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        self.reward=0
        cv2.imshow("Enduro", self._image)
        cv2.waitKey(1)
    def buildstate(self, grid):
        state = [0, 0]

        # Agent position (assumes the agent is always on row 0)
        [[x]] = np.argwhere(grid[0, :] == 2)
        state[0] = x

        # Sum the rows of the grid
        rows = np.sum(grid, axis=1)
        # Ignore the agent
        rows[0] -= 2
        # Get the closest row where an opponent is present
        rows = np.sort(np.argwhere(rows > 0).flatten())

        # If any opponent is present
        if rows.size > 0:
            # Add the x position of the first opponent on the closest row
            row = rows[0]
            for i, g in enumerate(grid[row, :]):
                if g == 1:
                    # 0 means that no agent is present and so
                    # the index is offset by 1
                    state[1] = i + 1
                    break
        return state
    def buildfeatures(self,state,action,grid):
        features=np.zeros((num_features))
        agent_x=state[3][0] #agent x coordinate in env grid
        opponment_x = state[3][1] #opponment x coordinate in env grid
        cars=state[1]
        agent= cars['self']
        others=cars['others']
        speed= state[2]
        leftwall= state[0][11][0][0]
        rightwall=state[0][11][10][0]
        if agent_x == 4 or agent_x ==5:
            features[0]=1
        else:
            features[0]=0
        if action== Action.ACCELERATE:
            if len(others) ==0:
                features[1]=1
            for i in range(len(others)):
                car_x= others[i][0]
                car_y= others[i][1]
                if car_x != agent[0] and agent_x != opponment_x:
                    features[2]= 1
                if car_x== agent[0] and abs(car_y-agent[1])>(2*agent[3]):
                    features[3]=1
            if speed <50:
                features[4]=1
            if agent_x+1!=opponment_x and agent_x-1 != opponment_x and agent_x != opponment_x:
                features[5]=1
            if speed == -50:
                features[6]=1
                # if agent[0] == car_x and speed!=-50:
                #     features[4]
        if action == Action.LEFT:
            for j in range(len(others)):
                car_x=others[j][0]
                car_y=others[j][1]
                if agent_x-1 != opponment_x and agent[0]-leftwall>agent[2]and agent_x == opponment_x:
                    features[7]=1
                else:
                    features[7]=0
        if action == Action.RIGHT:
            for k in range(len(others)):
                car_x=others[k][0]
                car_y=others[k][1]
                if agent_x+1 != opponment_x and agent[0]+agent[2]< rightwall and agent_x==opponment_x:
                    features[8]=1
                else:
                    features[8]=0
        if action == Action.BRAKE:
            for u in range(len(others)):
                car_x=others[u][0]
                car_y=others[u][1]
                if agent_x==opponment_x and abs(agent[1]-car_y)<agent[1] and agent_x-1==opponment_x and agent_x+1==opponment_x:
                    features[9]=1
                else:
                    features[9]=0
        return features
if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=2000, draw=True)
    print 'Total reward: ' + str(a.total_reward)
    print a.total_reward_list
    print("mean is:{0}".format(np.mean(a.total_reward_list)))
    print("variance is:{0}.".format(np.var(a.total_reward_list)))
    print a.weights_list
    fig_1 = plt.figure(figsize=(8,4))
    ax_1 = fig_1.add_subplot(111)
    # ax_2 = fig_1.add_subplot(1,2,2)
    # p1 = plt.subplot(211)
    # p2 = plt.subplot(212)
    # p3 = plt.subplot(213)
    ax_1.plot(range(0,a.episodes), a.total_reward_list)
    ax_1.set_xlabel("episode")
    ax_1.set_ylabel("total reward")
    ax_1.set_title("reward curve")
    plt.show()
    # ax_2.hist(a.total_reward_list)
    # ax_2.set_xlabel("total reward")
    # ax_2.set_ylabel("number")
    # ax_2.set_title

    for i in range(len(a.weights_list[0])):
        # fig_2= plt.figure(figsize=(8,4))
        # ax_3=fig_2.add_subplot(111)
        # ax_3.plot(range(1,a.episodes+1),a.weights_list[o])
        # ax_3.set_xlabel("episode")
        # ax_3.set_ylabel("weights")
        # ax_3.set_title("weights curve")
        p1=plt.subplot(211)
        temp=[]
        for me in range(a.episodes):
            temp.append(a.weights_list[me][i])
        p1.plot(range(0,a.episodes),temp)
        p1.set_xlabel("episode")
        p1.set_ylabel("weights")
        plt.show()
    # abs= './weights1.npy'
    # with open(abs,'wb') as f:
    #     np.save(f,a.weights_list)
    #     f.close()
    # abs1='./rewards1.npy'
    # with open(abs1,'wb') as f:
    #     np.save(f,a.total_reward_list)
    #     f.close()


