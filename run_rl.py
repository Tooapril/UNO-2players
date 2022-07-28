''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent
    agents = []
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        for _ in range(env.num_players):
            agent = DQNAgent(num_actions=env.num_actions,
                         state_shape=env.state_shape[0],
                         mlp_layers=[64,64],
                         device=device)
            agents.append(agent)
        env.set_agents(agents)
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        for _ in range(env.num_players):
            agent = NFSPAgent(num_actions=env.num_actions,
                          state_shape=env.state_shape[0],
                          hidden_layers_sizes=[64,64],
                          q_mlp_layers=[64,64],
                          device=device)
            agents.append(agent)
        env.set_agents(agents)
    elif args.algorithm == 'dmc':
        from rlcard.agents.dmc_agent import DMCTrainer
        # Initialize the DMC trainer
        trainer = DMCTrainer(env,
                            load_model=args.load_model,
                            log_dir=args.log_dir,
                            save_interval=args.save_interval,
                            num_actor_devices=args.num_actor_devices,
                            num_actors=args.num_actors,
                            training_device=args.training_device)
        
    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes): # 总共运行 num_episodes 局游戏

            if args.algorithm == 'dmc':
                trainer.start()
            else:
                if args.algorithm == 'nfsp':
                    agents[0].sample_episode_policy()

                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                # Here, we assume that DQN always plays the first position
                # and the other players play randomly (if any)
                for ts in trajectories[0]:
                    agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            # 每 evaluate_every 次进行一次评估，评估采取 num_eval_games 局游戏
            if episode % args.evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(env, args.num_eval_games)[0]) # 取玩家一的评估结果进行输出

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='dmc', 
                        choices=['dqn', 'nfsp', 'dmc'])
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=500000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=10000)
    parser.add_argument('--load_model', action='store_true',
                        help='Load an existing model')
    parser.add_argument('--log_dir', type=str, default='experiments/uno/dmc',
                        help='Root dir where experiment data will be saved')
    parser.add_argument('--save_interval', default=30, type=int,
                        help='Time interval (in minutes) at which to save the model')
    parser.add_argument('--num_actor_devices', default=1, type=int,
                        help='The number of devices used for simulation')
    parser.add_argument('--num_actors', default=3, type=int,
                        help='The number of actors for each simulation device')
    parser.add_argument('--training_device', default=0, type=int,
                        help='The index of the GPU used for training models')


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

