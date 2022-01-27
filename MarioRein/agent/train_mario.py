import warnings
from Policy import *
from Agent import DQNAgent
from Wrappers import make_env

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    env = make_env('SuperMarioBros-1-1-v1')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    save_directory = "mario_dqn_model"
    load_checkpoint = "checkpoint_2100.pth"
    time_to_save_model = 70
    ddqn_agent = DQNAgent(state_dim, env.action_space.n, EGreedyStategy(), save_directory)
    '''if load_checkpoint is not None:
        ddqn_agent.load_checkpoint(save_directory+"/"+load_checkpoint)'''
    dqn_softmax_result = ddqn_agent.train(env, 0.9, "DDQN", 2100, 1, time_to_save_model)
    # ddqn_agent.evaluate(ddqn_agent.online_model, env,100)
