from examples.gym.utils import run


if __name__ == '__main__':
    run('CartPole-v0', best_replay=True, search_type='evolutionary')
