from examples.gym.utils import run


if __name__ == '__main__':
    run('MountainCar-v0', generations=10, population_size=100, extra_neurons=4, best_replay=True, search_type='evolutionary', reset_state=False)
