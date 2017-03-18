import cv2
import numpy as np

from enduro.agent import Agent
from enduro.action import Action


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.current_reward = 0
        # feature vector generated from state-aciton pair
        self.feature = np.ones([10,])
        # Weight vector for each feature
        self.weight = np.zeros_like(self.feature)

        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}
        self.act2idx = {a: i for i, a in enumerate(self.getActionsSet())}


        # Learning rate
        self.ALPHA = 0.01
        # Discounting factor
        self.GAMMA = 0.9
        # Exploration rate
        self.EPS = 0.01

        # Log the obtained reward during learning
        self.last_episode = 1
        self.episode_log = np.zeros(6510) - 1.
        self.log = []


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
        self.weight += 0.1
        # Reset the total reward for the episode
        self.total_reward = 0
        self.next_state = self.buildState(road, cars, speed, grid):

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # self.total_reward += self.move(Action.ACCELERATE)
        self.state = self.next_state

        # If exploring
        if np.random.uniform(0., 1.) < self.epsilon:
            # Select a random action using softmax
            f_a = [self.buildFeature(self.state, action) for action in np.arange(4)]

            Q_s = [self.getQsa(feature, self.weight) for feature in f_a]
            probs = np.exp(Q_s) / np.sum(np.exp(Q_s))
            idx = np.random.choice(4, p=probs)
            self.action = self.idx2act[idx]
        else:
            # Select the greedy action
            self.action = self.idx2act[self.argmaxQsa(self.state)]

        # After selecting action, the feature can be computed

        self.feature = self.buildFeature(self.state, self.act2idx[self.action])

        self.reward = self.move(self.action)
        self.total_reward += self.reward

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
        print grid
        # print road[11][0]
        # print road[11][10]
        # print cars
        self.next_state = self.buildState(road, cars, speed, grid)

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        Q_sa = self.getQsa(self.feature, self.weight)

        # Calculate the error
        error = self.reward + self.GAMMA * self.maxQsa(self.next_state) - Q_sa

        # Update the weights
        self.weight = self.weight + self.ALPHA * error * self.feature


    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        if not iteration % 1000:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)

        # Initialise the log for the next episode
        if episode != self.last_episode:
            iters = np.nonzero(self.episode_log >= 0)
            rewards = self.episode_log[iters]
            self.log.append((np.asarray(iters).flatten(), rewards, np.copy(self.Q)))
            self.last_episode = episode
            self.episode_log = np.zeros(6510) - 1.

        # Log the reward at the current iteration
        self.episode_log[iteration] = self.total_reward

        if not episode % 100:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(20)

    def buildState(self, road, cars, speed, grid):
        # each component in the state vector is:
        state = []
        # 0 (int in [-50, 50]): speed of agent
        state.append(speed)

        # 1 (int in [0,9]): position of agent
        [[x]] = np.argwhere(grid[0, :] == 2)
        state.append(x)

        # 2 (Boolean): If Collision occurred
        state.append(self.collision(cars))

        # 3 (int): direction of road 2 for very right,
        # -2 for very left, 0 for straight
        if road[0][0] - 105 > 20:
            state.append[2]
        elif road[0][0] - 105 > 5:
            state.append[1]
        elif road[0][0] - 105 > -8:
            state.append[0]
        elif road[0][0] -105 > -30:
            state.append[-1]
        else:
            state.append[-2]

        # 4,5,6 sensor_0, sensor_90_all, sensor_180
        agent_pt = np.argmax(grid)
        sensor_0 = agent_pt
        for i in range(agent_pt):
            if grid[0,i] == 1:
                sensor_0 -= i
                break

        if sensor_0 > 2:
            sensor_0 = 1
        else:
            sensor_0 = 0

        sensor_180 = 9-agent_pt
        for i in range(9-agent_pt):
            if grid[0,i+agent_pt] ==1:
                sensor_180 = i
                break
        if sensor_180 > 2:
            sensor_180 = 1
        else:
            sensor_180 = 0

        sensor_90 = 10
        sensor_90_l = 10
        sensor_90_r = 10

        if agent_pt <= 1:
            sensor_90_l = 10
        else:
            for i in range(10):
                if grid[i, agent_pt-1] == 1:
                    sensor_90_l = i
                    break

        if agent_pt >= 8:
            sensor_90_r = 10
        else:
            for i in range(10):
                if grid[i, agent_pt+1] == 1:
                    sensor_90_r = i
                    break

        for i in range(10):
            if grid[i, agent_pt] == 1:
                sensor_90 = i
                break

        if sensor_90 > 5:
            sensor_90 = 1
        else:
            sensor_90 = 0

        if sensor_90_l > 5:
            sensor_90_l = 1
        else:
            sensor_90_l = 0

        if sensor_90_r > 5:
            sensor_90_r = 1
        else:
            sensor_90_r =0

        sensor_90_all = sensor_90 * sensor_90_l * sensor_90_r
        # # 4 (int): minist distance of opponent
        # # 5 (int): minist angle
        # if not cars['others']:
        #     min_dist = None
        #     min_angle = None
        # else:
        #     x, y, _, _ = cars['self']
        #     min_dist = sys.float_info.max
        #     min_angle = 0.
        #
        #     for c in cars['others']:
        #         cx, cy, _, _ = c
        #         dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
        #         if dist < min_dist:
        #             min_dist = dist
        #             min_angle = np.arctan2(y - cy, cx - x)
        # state.append[min_dist]
        # state.append[min_angle]

        return state

    def buildFeature(self, state, action):
        feature = np.ones([10,])


        return feature

    def getQsa(self, feature, weight):
        return np.dot(weight,feature)

    def maxQsa(self, state):
        return np.max(self.Q[state[0], state[1], :])

    def argmaxQsa(self, state):
        return np.argmax(self.Q[state[0], state[1], :])

if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=2000, draw=True)
    print 'Total reward: ' + str(a.total_reward)
