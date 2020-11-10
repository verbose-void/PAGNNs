from examples.gym.utils import run, deterministic


if __name__ == '__main__':
    deterministic()
    run('Pong-ram-v0', generations=100, population_size=100, extra_neurons=0, best_replay=True, search_type='evolutionary', retain_state=False)
