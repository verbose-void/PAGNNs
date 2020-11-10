import gym
from pagnn.pagnn import PAGNNLayer

def play_episode(network, episodes=20, max_steps=100, render=False, verbose=False):
    env = gym.make('CartPole-v0')
    scores = []

    for i_episode in range(episodes):
        observation = env.reset()
        score = 0
        for t in range(max_steps):
            if render:
                env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                if verbose:
                    print("Episode finished after {} timesteps, score: {}".format(t+1, score))

                break

        scores.append(score)

    if render:
        env.close()

    return scores


if __name__ == '__main__':
    pagnn = PAGNNLayer(4, 1, 5)
    scores_per_episode = play_episode(pagnn)
    print(scores_per_episode)