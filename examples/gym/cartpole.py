from examples.gym.utils import run, deterministic


if __name__ == '__main__':
    deterministic()
    run('CartPole-v0', generations=10, population_size=100, extra_neurons=0, best_replay=True, search_type='evolutionary')
