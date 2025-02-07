################################################################################################################
# Authors:                                                                                                     #
# Hongyao Tang                                                                            #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://github.com/kenjyoung/MinAtar                                                                 #
################################################################################################################

import time, argparse
import numpy as np
from minatar import Environment


def random_action(num_actions):
    return np.random.choice([i for i in range(num_actions)])


def get_args_from_parser():
    parser = argparse.ArgumentParser()

    # breakout, freeway, asterix, seaquest, space_invaders
    parser.add_argument("--env", type=str, default='space_invaders', help='env')
    parser.add_argument("--seed", type=int, default=1, help='random_seed')
    parser.add_argument("--max-step", type=int, default=10, help='total_steps(k)')
    parser.add_argument("--gpu-no", type=str, default='-1', help='gpu_no')

    parser.add_argument("--lr", type=float, default=0.0001, help='learning_rate')
    parser.add_argument("--ti", type=int, default=5, help='train_interval')

    parser.add_argument("--eval-interval", type=int, default=20000, help='number of steps per evaluation point')
    parser.add_argument("--is-save-data", action="store_true", help='is_save_data')

    return parser.parse_args()


def run_exp(args):
    #####################  hyper parameters  ####################

    MAX_TOTAL_STEPS = 1000 * args.max_step

    env = Environment(args.env)
    env.seed(args.seed)
    # Get channels and number of actions specific to each game
    state_shape = env.state_shape()
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    print(env.reset())

    print('-- Env:', args.env)
    print('-- Seed:', args.seed)
    print('-- Configurations:', args)

    return_his = []
    global_step_count = 0
    ep_num = 0

    while global_step_count < MAX_TOTAL_STEPS:
        env.reset()
        s = env.state()
        done = False

        ep_reward = 0
        ep_step_count = 0

        while (not done) and global_step_count < MAX_TOTAL_STEPS:
            # Interact with env
            a = random_action(num_actions)
            r, is_terminated = env.act(a)
            s_ = env.state()

            ep_step_count += 1
            global_step_count += 1

            s = s_
            ep_reward += r

        ep_num += 1
        return_his.append(ep_reward)

        print('- Steps:', global_step_count, '/', MAX_TOTAL_STEPS, 'Ep:', ep_num, ', return:', ep_reward, 'ep_steps:', ep_step_count)


if __name__ == '__main__':

    t1 = time.time()
    arguments = get_args_from_parser()
    run_exp(args=arguments)
    print('Running time: ', time.time() - t1)
