import cv2
import numpy as np

from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import sys

class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0

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

        # Reset the total reward for the episode
        self.total_reward = 0

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        key = cv2.waitKey(0)
        action = Action.NOOP
        if chr(key & 255) == 'a':
            action = Action.LEFT
        if chr(key & 255) == 'd':
            action = Action.RIGHT
        if chr(key & 255) == 'w':
            action = Action.ACCELERATE
        if chr(key & 255) == 's':
            action = Action.BRAKE

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT or Action.ACCELERATE
        # Do not use plain integers between 0 - 3 as it will not work
        self.total_reward += self.move(action)

        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work
        pass

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
        print (road[0][5][0]- road[11][5][0])
        x, _, _, _ = cars['self']
        print x
        # if cars['others']:
        #     x, y, _, _ = cars['self']
        #
        #     min_dist = sys.float_info.max
        #     min_angle = 0.
        #
        #     for c in cars['others']:
        #         cx, cy, _, _ = c
        #         dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
        #         if dist < min_dist:
        #             min_dist = dist
        #             min_angle = np.arctan2(y - cy, cx - x)
        #     print min_dist
        #     print 180*(min_angle/np.pi)
        # cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)

        # You could comment this out in order to speed up iterations
        cv2.imshow("Enduro", self._image)
        cv2.waitKey(40)


if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=2000, draw=True)
    print 'Total reward: ' + str(a.total_reward)
