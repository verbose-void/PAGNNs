from examples.gym.utils import run, deterministic


if __name__ == '__main__':
    deterministic()
    run('MountainCar-v0', generations=10, population_size=100, extra_neurons=3, best_replay=True, search_type='evolutionary', retain_state=False)
