import cv2
import numpy as np
import pickle

from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.reward = 0
        # feature vector generated from state-aciton pair
        self.feature = np.ones([12,])
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
        self.EPSILON = 0.01

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
        self.weight[1:5] -=0.2
        # Make it quicker to accelerate at first
        self.weight[5] += 1.
        self.weight[7] += 1.
        # Reset the total reward for the episode
        self.total_reward = 0
        self.next_state = self.buildState(road, cars, speed, grid)

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        self.state = self.next_state

        # If exploring
        if np.random.uniform(0., 1.) < self.EPSILON:
            # Select a random action using softmax
            # f_a = [self.buildFeature(self.state, action) for action in np.arange(4)]
            # Q_s = [self.getQsa(feature, self.weight) for feature in f_a]
            # probs = np.exp(Q_s) / np.sum(np.exp(Q_s))
            # idx = np.random.choice(4, p=probs)
            idx = np.random.choice(4)
            self.action = self.idx2act[idx]
        else:
            # Select the greedy action
            self.action = self.idx2act[self.argmaxQsa(self.state)]

        # After selecting action, the feature can be computed

        self.feature = self.buildFeature(self.state, self.act2idx[self.action])

        self.reward = self.move(self.action)
        self.total_reward += self.reward

        # key = cv2.waitKey(0)
        # self.action = Action.NOOP
        # if chr(key & 255) == 'a':
        #     action = Action.LEFT
        # if chr(key & 255) == 'd':
        #     action = Action.RIGHT
        # if chr(key & 255) == 'w':
        #     action = Action.ACCELERATE
        # if chr(key & 255) == 's':
        #     action = Action.BRAKE
        # self.action = action
        # self.reward = self.move(action)
        # self.total_reward += self.reward
        # self.feature = self.buildFeature(self.state, self.act2idx[self.action])

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

        self.next_state = self.buildState(road, cars, speed, grid)

        # Visualise the environment grid
        # cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

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
        # print episode
        # print "state: speed: {0}, position: {1} \n danger: {2} {3} {4}\n Road: {5}".format(
        #         self.state[0],self.state[1],self.state[2],self.state[3],self.state[4],self.state[6])
        # print "action: {}".format(self.act2idx[self.action])
        if not iteration % 1000:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
            print self.weight

        # Initialise the log for the next episode
        if episode != self.last_episode:
            iters = np.nonzero(self.episode_log >= 0)
            rewards = self.episode_log[iters]
            self.log.append((np.asarray(iters).flatten(),rewards, np.copy(self.weight)))
            self.last_episode = episode
            self.episode_log = np.zeros(6510) - 1.

        # Log the reward at the current iteration
        self.episode_log[iteration] = self.total_reward
        # if not episode % 1:
        cv2.imshow("Enduro", self._image)
        # q= cv2.waitKey(0)
        # if q == 'ESC':
            # pass

    def buildState(self, road, cars, speed, grid):
        # each component in the state vector is:
        state = np.zeros([7,])
        # 0 (int in [-50, 50]): speed of agent
        state[0] = speed

        # 1 (int in [0,9]): position of agent
        [[x]] = np.argwhere(grid[0, :] == 2)
        # state.append(x)
        state[1] = x

        # 2-4: the dangerous in left, right and front direction
        # Dangerous is measured by the number of opponent cars
        # in the corresponding alert areas
        left_grid = grid[:4,:x]
        right_grid = grid[:4,x+1:]
        front_grid = grid[1:5,x]
        if left_grid.shape[1] > 2:
            left_grid = left_grid[:,-2:]
            # right_grid[-1,0] = 0
        state[2] = np.sum(left_grid)
        if right_grid.shape[1] > 2:
            right_grid = right_grid[:,:2]
            # left_grid[-1,-1]=0

        state[3] = np.sum(right_grid)
        state[4] = np.sum(front_grid)

        # 5 (Boolean): If Collision occurred
        state[5] = self.collision(cars)

        # 6 (int in [-2,2]): Deviation of the road
        # it is positive if it turns right
        # it is negative if it turn left
        state[6] = road[0][5][0]- road[11][5][0]

        return state

    def buildFeature(self, state, action):
        feature = np.zeros([12,])
        feature[0] = 1.0 # The biase
        #In dangerous in front direction And Speed > 0,
        # action a will lead to a collision
        if state[0] > 0 and action==0 and state[4] > 0:
            feature[1] = 1.0
        if state[0] > 0 and action==1 and state[3] > 0:
            feature[2] =1.0
        if state[0] > 0 and action==2 and state[2] > 0:
            feature[3] = 1.0
        if state[5]:
            feature[4] = 1.0
        if state[4] == 0 and action==0:
            feature[5] = 1.0
        feature[6] = (state[0]+50)/100.0
        if state[0] < 40 and action==0:
            feature[7] = 1.0
        if state[1] < 4 and action==1:
            feature[8] = 1.0
        if state[1] > 5 and action==2:
            feature[9] = 1.0
        if state[6] > 10 and action==2:
            feature[10] = 1.0
        if state[6] < -10 and action==1:
            feature[11] = 1.0
        return feature

    def getQsa(self, feature, weight):
        Q =  np.dot(weight,feature)
        return Q

    def maxQsa(self, state):
        return np.max([self.getQsa(self.buildFeature(state, action), self.weight)
                    for action in np.arange(4)])
        # return np.max(self.Q[state[0], state[1], :])

    def argmaxQsa(self, state):
        # return np.argmax(self.Q[state[0], state[1], :])
        return np.argmax([self.getQsa(self.buildFeature(state, action), self.weight)
                    for action in np.arange(4)])

if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=2000, draw=True)
    print 'Total reward: ' + str(a.total_reward)
    print "weight: {}".format(a.weight)
    pickle.dump(a.log, open("log.p", "wb"))
