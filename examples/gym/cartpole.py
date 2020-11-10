from examples.gym.utils import run, set_seed_reproducability


if __name__ == '__main__':
    set_seed_reproducability()
    run('CartPole-v0', best_replay=True, search_type='evolutionary')
