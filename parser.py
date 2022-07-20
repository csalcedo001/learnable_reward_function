from argparse import ArgumentParser

from utils import get_config_from_string

def get_parser(episodes=300, max_iter=1000, num_samples=1):
    parser = ArgumentParser()

    parser.add_argument(
        'agent',
        type=str)
    parser.add_argument(
        '--env',
        type=str, default='CartPole-v1')
    parser.add_argument(
        '--env-config',
        type=get_config_from_string, default={})
    parser.add_argument(
        '--agent-config',
        type=get_config_from_string, default={})
    parser.add_argument(
        '--checkpoint',
        type=str, default=None)
    parser.add_argument(
        '--episodes',
        type=int, default=episodes)
    parser.add_argument(
        '--seed',
        type=int, default=0)
    parser.add_argument(
        '--num-samples',
        type=int, default=num_samples)
    parser.add_argument(
        '--max-iter',
        type=int, default=max_iter)
    parser.add_argument(
        '--no-render',
        default=False, action='store_const', const=True)

    return parser