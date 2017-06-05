from ale_python_interface import ALEInterface
from enduro.action import Action
from enduro.control import Controller
from enduro.state import StateExtractor


class Agent(object):
    def __init__(self):
        self._ale = ALEInterface()
        self._ale.setInt('random_seed', 123)
        self._ale.setFloat('repeat_action_probability', 0.0)
        self._ale.setBool('color_averaging', False)
        self._ale.loadROM('roms/enduro.bin')
        self._controller = Controller(self._ale)
        self._extractor = StateExtractor(self._ale)
        self._image = None
        self.epoch_number=100

    def run_Q(self, learn, episodes=1, draw=False):
        """ Implements the playing/learning loop.

        Args:
            learn(bool): Whether the self.learn() function should be called.
            episodes (int): The number of episodes to run the agent for.
            draw (bool): Whether to overlay the environment state on the frame.

        Returns:
            None
        """
        for e in range(episodes):
            # Observe the environment to set the initial state
            (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)
            self.initialise(grid)
            # for i in range(1,9):
            #     if grid[0,i]==2:
            #         agent_location=i
            #         break
            #ob=grid[:,agent_location-1:agent_location+2]
            num_frames = self._ale.getFrameNumber()

            # Each episode lasts 6500 frames
            while self._ale.getFrameNumber() - num_frames < 6500:
                # Take an action
                action= self.act()

                # Update the environment grid
                (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)

                # for j in range(1,9):
                #     if grid[0,j]==2:
                #         agent_location=j
                #         break
                #ob_=grid[:,agent_location-1:agent_location+2]

                self.sense(grid)

                # Perform learning if required
                if (learn):
                    self.learn(action)
                #ob=ob_
                self.callback(learn, e + 1, self._ale.getFrameNumber() - num_frames)
            self._ale.reset_game()
    def run_Q_new(self, learn, episodes=1, draw=False):
        """ Implements the playing/learning loop.

        Args:
            learn(bool): Whether the self.learn() function should be called.
            episodes (int): The number of episodes to run the agent for.
            draw (bool): Whether to overlay the environment state on the frame.

        Returns:
            None
        """
        for e in range(episodes):
            # Observe the environment to set the initial state
            (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)
            self.initialise(grid)
            for i in range(1,9):
                if grid[0,i]==2:
                    agent_location=i
                    break
            ob=grid[0:5,agent_location-1:agent_location+2]
            num_frames = self._ale.getFrameNumber()

            # Each episode lasts 6500 frames
            while self._ale.getFrameNumber() - num_frames < 6500:
                # Take an action
                action= self.act(str(ob))

                # Update the environment grid
                (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)

                for j in range(1,9):
                    if grid[0,j]==2:
                        agent_location=j
                        break
                ob_=grid[0:5,agent_location-1:agent_location+2]

                self.sense(grid)

                # Perform learning if required
                if (learn):
                    self.learn(action,str(ob),str(ob_))
                    ob=ob_
                self.callback(learn, e + 1, self._ale.getFrameNumber() - num_frames)
            self._ale.reset_game()

    def run_test(self, learn, episodes=1, draw=False):
        """ Implements the playing/learning loop.

        Args:
            learn(bool): Whether the self.learn() function should be called.
            episodes (int): The number of episodes to run the agent for.
            draw (bool): Whether to overlay the environment state on the frame.

        Returns:
            None
        """

        for e in range(episodes):
            # Observe the environment to set the initial state
            (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)
            self.initialise(grid)

            num_frames = self._ale.getFrameNumber()

            # Each episode lasts 6500 frames
            while self._ale.getFrameNumber() - num_frames < 6500:
                action= self.act()
                # Update the environment grid
                (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)
                self.sense(grid)
                # Perform learning if required
                if (learn):
                    self.learn()
                self.callback(learn, e + 1, self._ale.getFrameNumber() - num_frames)
            self._ale.reset_game()

    def getActionsSet(self):
        """ Returns the set of all possible actions
        """
        return [Action.ACCELERATE, Action.RIGHT, Action.LEFT, Action.BREAK]

    def move(self, action):
        """ Executes the action and advances the game to the next state.

        Args:
            action (int): The action which should executed. Make sure to use
                          the constants returned by self.getActionsSet()

        Returns:
           int: The obtained reward after executing the action
        """
        return self._controller.move(action)

    def initialise(self, grid):
        """ Called at the beginning of each episode, mainly used
        for state initialisation.

        Args:
            grid (np.ndarray): 11x10 array with the initial environment grid.

        Returns:
            None
        """
        raise NotImplementedError

    def act(self):
        """ Called at each loop iteration to choose and execute an action.

        Returns:
            None
        """
        raise NotImplementedError

    def sense(self, grid):
        """ Called at each loop iteration to construct the new state from
        the update environment grid.

        Returns:
            None
        """
        raise NotImplementedError

    def learn(self):
        """ Called at each loop iteration when the agent is learning. It should
        implement the learning procedure.

        Returns:
            None
        """
        raise NotImplementedError

    def callback(self, learn, episode, iteration):
        """ Called at each loop iteration mainly for reporting purposes.

        Args:
            learn (bool): Indicates whether the agent is learning or not.
            episode (int): The number of the current episode.
            iteration (int): The number of the current iteration.

        Returns:
            None
        """

        raise NotImplementedError
