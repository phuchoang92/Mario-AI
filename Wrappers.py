import cv2
import gym
import collections
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT


class PreprocessFrames(gym.ObservationWrapper):
    """
    PREPROCESSES EACH FRAME (input = (rows, columns, 3)) [0-255]
    1. resize image                             -   (new_rows, new_columns, 1)       [0-255]
    2. convert to nparray                       -   array(new_rows, new_columns, )  [0-255]
    3. move axis(reshape)                       -   array(1, new_rows, new_columns)  [0-255]
    3. scale values from 0-1                    -   array(1, new_rows, new_columns)  [0.0-1.0]
    """

    def __init__(self, env, new_observation_shape):
        super().__init__(env)
        self.new_observation_shape = new_observation_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.new_observation_shape, dtype=np.float32)

    def observation(self, observation):
        temp_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        temp_frame = cv2.resize(temp_frame, self.new_observation_shape[1:], interpolation=cv2.INTER_AREA)
        new_observation = np.array(temp_frame, dtype=np.float32).reshape(self.new_observation_shape)
        new_observation = new_observation / 255.0
        return new_observation


# TO BE CALLED ON EACH SINGLE IMAGE (AFTER PREPROCESS)
class CustomStep(gym.Wrapper):
    """
    OVERRIDES step() & reset()
    1. repeats same action in 'n' skipped frames to compute faster.
    2. takes maximum of 2 frames.
    """

    def __init__(self, env, frame_skip):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.observation_shape = env.observation_space.shape
        self.observation_buffer = np.zeros_like((2, self.observation_shape))

    def reset(self):
        observation = self.env.reset()
        self.observation_buffer = np.zeros_like((2, self.observation_shape))
        self.observation_buffer[0] = observation
        return observation

    # RETURN FRAME_SKIPPED FRAMES
    def step(self, action):
        total_reward = 0.0
        done = False

        for frame in range(self.frame_skip):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward

            idx = frame % 2
            self.observation_buffer[idx] = observation

            if done: break

        observation_max = np.maximum(self.observation_buffer[0], self.observation_buffer[1])
        return observation_max, total_reward, done, info


# STACK OBSERVATIONS
class StackFrames(gym.ObservationWrapper):
    """
    STACKS stack_size FRAMES TOGETHER AND RETURNS AS THE 'observation'
    1. on reset() returns first 'observation' STACKED 'stack_size' times
    2. observation() returns current 'observation' STACKED with 'stack_size-1' previous 'observation'
    """

    def __init__(self, env, stack_size):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(stack_size, axis=0),
            env.observation_space.high.repeat(stack_size, axis=0)
        )
        self.stack = collections.deque(maxlen=stack_size)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        observation = np.array(self.stack).reshape(self.observation_space.shape)
        return observation

    def observation(self, observation):
        self.stack.append(observation)
        observation = np.array(self.stack).reshape(self.observation_space.shape)
        return observation


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._current_score = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return observation, reward / 10.0, done, info


def make_env(env_name, new_observation_shape=(1, 84, 84), stack_size=4, frame_skip=4):
    env = gym.make(env_name)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = PreprocessFrames(env, new_observation_shape=new_observation_shape)
    env = CustomStep(env, frame_skip=4)
    env = StackFrames(env, stack_size=stack_size)
    env = CustomReward(env)
    return env
