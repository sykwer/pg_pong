import numpy as np
import _pickle as pickle
import gym

from pg_agent import PGAgent
from const import GAME_NAME

BATCH_SIZE = 10
SAVE_SIZE = 100
SHOW_MOVIE = True

def main():
    env = gym.make(GAME_NAME)
    agent = PGAgent(env)
    observation = env.reset()

    episodes_num = 0
    reward_sum = 0

    while True:
        if SHOW_MOVIE: env.render()

        action = agent.select_action(observation)
        observation, reward, done, info = env.step(action) # move paddle!!!1!

        reward_sum += reward
        agent.add_history(reward)

        if reward != 0: # when either player gets point
            print("Episode %d fin. Gets reward %f" % (episodes_num, reward))

        if done: # when episode ends
            episodes_num += 1
            agent.accumulate_grads()
            if episodes_num % BATCH_SIZE == 0: agent.train_net()
            if episodes_num % SAVE_SIZE == 0: pickle.dump(agent.net.model, open("model.p", "wb"))

            print("Reset env. Total rewards in this episode: %f" % reward_sum)

            reward_sum = 0
            observation = env.reset()
            agent.prev_x = None


if __name__ == "__main__":
    main()
