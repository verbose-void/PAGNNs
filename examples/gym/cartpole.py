import gym
import numpy as np
import torch
from pagnn.pagnn import PAGNNLayer
from copy import deepcopy


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def play_episodes(network, episodes=20, max_steps=5000, render=False, verbose=False):
    env = gym.make('CartPole-v0')
    scores = []

    for i_episode in range(episodes):
        observation = env.reset()
        score = 0
        for t in range(max_steps):
            if render:
                env.render()

            x = torch.tensor(observation, device=device)
            y = network(x)
            action = torch.argmax(y).item()
            
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


def genome_generator():
    return PAGNNLayer(4, 2, 4).to(device)


def get_random_genomes(n):
    return [genome_generator() for _ in range(n)]


def run(generations=1, population_size=100, best_replay=False):
    best_genome = {'score': float('-inf')}

    for generation in range(generations):
        avg_scores_per_genome  = []
        genomes = get_random_genomes(population_size)

        # let genomes play
        for i, genome in enumerate(genomes):
            scores_per_episode = play_episodes(genome)
            avg_score = np.mean(scores_per_episode).item()
            avg_scores_per_genome.append(avg_score)

            if avg_score > best_genome['score']:
                best_genome = {'score': avg_score, 'state_dict': deepcopy(genome.state_dict())}

        print('avg scores per genome:', avg_scores_per_genome)

    print('best score:', best_genome['score'])
    
    # replay
    if best_replay:
        genome = genome_generator()
        genome.load_state_dict(best_genome['state_dict'])
        play_episodes(genome, verbose=True, render=True, episodes=1, max_steps=5000)


if __name__ == '__main__':
    run(best_replay=True)
