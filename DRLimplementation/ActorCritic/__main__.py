# coding=utf-8
"""
Invoke Actor-Critic agent using

    python -m ActorCritic [--help] [--train] [--render_training] [--discounted] [--play_for] [--test_run]

"""


import argparse

parser = argparse.ArgumentParser(description=(
    "=============================================================================\n"
    ":: Command line option for the ActorCritic Agent.\n\n"
    "   The agent will play by default using previously trained computation graph.\n"
    "   You can execute training by using the argument: --train "),
    epilog="=============================================================================\n")


# parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--train', action='store_true',
                    help='Execute training of ActorCritic agent')

parser.add_argument('-r', '--render_training', action='store_true',
                    help='(Training option) Watch the agent execute trajectories while he is on traning duty')
parser.add_argument('-d', '--discounted', default=None, type=bool,
                    help='(Training option) Force training execution with discounted reward-to-go')

parser.add_argument('-p', '--play_for', type=int, default=20,
                    help='(Playing option) Max playing trajectory, default=20')
parser.add_argument('--test_run', action='store_true')

args = parser.parse_args()

if args.train:
    raise NotImplementedError   # todo: implement
    # train_ActorCritic_agent_discrete(render_env=args.render_training,
    #                                discounted_reward_to_go=args.discounted,
    #                                test_run=args.test_run)
else:
    raise NotImplementedError   # todo: implement
    # play_ActorCritic_agent_discrete(max_trajectories=args.play_for, test_run=args.test_run)



