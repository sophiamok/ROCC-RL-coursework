"""
Reproduction of Figure 4 of Treloar et al. (2020): Online Fitted Q-learning.

    python scripts/run_fig4.py --seed 0

A single agent controls five chemostats running in parallel, taking an
independent action for each every five minutes from one shared policy, and
refitting its Q-network every ten timesteps using all experience gathered so
far (Algorithm 3). Total duration is twenty-four hours of simulated time. The
agent is then held at zero exploration for a further 310 minutes with no
learning, as in Fig 4 and S5, to show the target behaviour is maintained.

This is a different algorithm from Fig 2: online rather than episodic, and the
environments are never reset, so each chemostat's trajectory is one continuous
24-hour run rather than a sequence of independent episodes.

Derived from examples/Figure_4/parallel_example.py, with seeding, the hold
phase, exact step accounting (288 steps, not 290) and provenance capture added.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

from ROCC import ChemostatEnv, KerasFittedQAgent, reward_func

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PARAMS = os.path.join('ROCC', 'chemostat_env', 'parameter_files', 'double_aux.yaml')

ONE_MIN = 0.016666666667
POP_SCALING = 100000


def git_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', REPO, 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def git_dirty_tracked():
    try:
        out = subprocess.check_output(
            ['git', '-C', REPO, 'status', '--porcelain', '--untracked-files=no'])
        return bool(out.strip())
    except Exception:
        return None


def cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n-envs', type=int, default=5)
    ap.add_argument('--sampling-minutes', type=int, default=5)
    ap.add_argument('--train-minutes', type=int, default=24 * 60,
                    help='online training duration, 1440 = 24 h')
    ap.add_argument('--hold-minutes', type=int, default=310,
                    help='post-training hold at explore=0, no learning (Fig 4, S5)')
    ap.add_argument('--update-every', type=int, default=10,
                    help='timesteps between value-function updates (Algorithm 3)')
    ap.add_argument('--params', default=DEFAULT_PARAMS)
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'raw'))
    args = ap.parse_args()

    param_file = args.params if os.path.isabs(args.params) else os.path.join(REPO, args.params)
    if not os.path.exists(param_file):
        raise SystemExit('no such parameter file: {}'.format(param_file))

    run_id = 'fig4_online_seed{}'.format(args.seed)
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    sampling_time = args.sampling_minutes * ONE_MIN
    total_steps = int(args.train_minutes / args.sampling_minutes)     # 288
    hold_steps = int(args.hold_minutes / args.sampling_minutes)       # 62

    # Chunk the 288 steps into blocks of `update_every`, with a short final
    # block rather than overshooting to 290 as the bundled example does.
    chunks = [args.update_every] * (total_steps // args.update_every)
    if total_steps % args.update_every:
        chunks.append(total_steps % args.update_every)
    n_chunks = len(chunks)

    envs = [ChemostatEnv(param_file, reward_func, sampling_time, POP_SCALING)
            for _ in range(args.n_envs)]
    agent = KerasFittedQAgent(
        layer_sizes=[envs[0].num_controlled_species, 20, 20,
                     envs[0].num_Cin_states ** envs[0].num_controlled_species],
        seed=args.seed,
    )
    agent.memory = []

    print('online NFQ: {} chemostats, {} steps of {} min = {:.1f} h training, '
          'update every {} steps ({} updates)'.format(
              args.n_envs, total_steps, args.sampling_minutes,
              args.train_minutes / 60.0, args.update_every, n_chunks))
    print('then {} steps = {} min hold at explore 0, no learning'.format(
        hold_steps, args.hold_minutes))
    print()

    chunk_returns = np.zeros((n_chunks, args.n_envs))
    explore_rates, chunk_seconds = [], []
    early_stop = [0] * args.n_envs
    t_start = time.time()
    step_cursor = 0

    for c, chunk_len in enumerate(chunks):
        t_c = time.time()
        explore_rate = agent.get_rate(c, 0.05, 1.0, n_chunks / 10.0)

        # Gather experience from the first n-1 chemostats without fitting, then
        # fit once on the last, which trains on the pooled memory of all five.
        for e, env in enumerate(envs):
            is_last = (e == args.n_envs - 1)
            traj, r = agent.run_episode(env, explore_rate, chunk_len,
                                        train=is_last, remember=True)
            chunk_returns[c, e] = r
            if len(agent.memory[-1]) < chunk_len:
                early_stop[e] += 1

        step_cursor += chunk_len
        explore_rates.append(explore_rate)
        chunk_seconds.append(time.time() - t_c)
        print('update {:3d}/{:d}  step {:3d}  explore {:.4f}  '
              'mean reward/step {:+.4f}  {:5.1f}s'.format(
                  c + 1, n_chunks, step_cursor, explore_rate,
                  chunk_returns[c].mean() / chunk_len, chunk_seconds[-1]), flush=True)

    train_seconds = time.time() - t_start

    # Hold phase: trained policy, no exploration, no learning, no memory writes.
    print('\nhold phase ({} steps, explore 0, no learning)'.format(hold_steps))
    hold_returns = np.zeros(args.n_envs)
    t_hold = time.time()
    for e, env in enumerate(envs):
        _, r = agent.run_episode(env, 0.0, hold_steps, train=False, remember=False)
        hold_returns[e] = r
    hold_seconds = time.time() - t_hold
    print('hold mean reward/step {:+.4f} (max 0.1)'.format(
        hold_returns.mean() / hold_steps))

    total_seconds = time.time() - t_start

    # Final populations, averaged over the hold phase, against the targets.
    final_pops = np.array([env.sSol[-hold_steps:, :2].mean(axis=0) for env in envs])

    for e, env in enumerate(envs):
        np.save(os.path.join(out_dir, 'trajectory_env{}.npy'.format(e)),
                np.array(env.sSol))
    np.save(os.path.join(out_dir, 'chunk_returns.npy'), chunk_returns)
    np.save(os.path.join(out_dir, 'hold_returns.npy'), hold_returns)
    np.save(os.path.join(out_dir, 'explore_rates.npy'), np.array(explore_rates))
    np.save(os.path.join(out_dir, 'chunk_seconds.npy'), np.array(chunk_seconds))
    np.save(os.path.join(out_dir, 'final_populations.npy'), final_pops)
    agent.save_network(out_dir)

    config = {
        'run_id': run_id,
        'experiment': 'online NFQ (Algorithm 3), Fig 4 + S5',
        'seed': args.seed,
        'n_envs': args.n_envs,
        'sampling_minutes': args.sampling_minutes,
        'train_minutes': args.train_minutes,
        'train_steps': total_steps,
        'hold_minutes': args.hold_minutes,
        'hold_steps': hold_steps,
        'update_every_steps': args.update_every,
        'n_updates': n_chunks,
        'discount_gamma': agent.gamma,
        'layer_sizes': agent.layer_sizes,
        'param_file': os.path.relpath(param_file, REPO),
        'yields_auxotrophic': [float(v) for v in np.ravel(envs[0].y)],
        'yields_carbon': [float(v) for v in np.ravel(envs[0].y0)],
        'targets_cells_per_L': [20e9, 30e9],
        'max_reward_per_step': 0.1,
        'chunks_terminated_early_per_env': early_stop,
        'mean_reward_per_step_final_update': float(
            chunk_returns[-1].mean() / chunks[-1]),
        'mean_reward_per_step_hold': float(hold_returns.mean() / hold_steps),
        'sd_reward_per_step_hold': float((hold_returns / hold_steps).std()),
        'mean_final_populations_1e9': (final_pops / 1000.0).round(3).tolist(),
        'train_seconds': round(train_seconds, 1),
        'hold_seconds': round(hold_seconds, 1),
        'total_seconds': round(total_seconds, 1),
        'git_commit': git_commit(),
        'git_dirty': git_dirty_tracked(),
        'python': sys.version.split()[0],
        'tensorflow': tf.__version__,
        'numpy': np.__version__,
        'platform': platform.platform(),
        'cpu': cpu_model(),
        'cpu_count': os.cpu_count(),
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print('\nwrote {}'.format(out_dir))
    print('mean populations over the hold phase (x10^9 cells/L), target [20, 30]:')
    for e, p in enumerate(final_pops / 1000.0):
        print('  chemostat {}: [{:6.2f}, {:6.2f}]'.format(e, p[0], p[1]))
    if any(early_stop):
        print('WARNING: washout detected, chunks ending early per env: {}'.format(early_stop))
    print('{:.1f} min total'.format(total_seconds / 60))


if __name__ == '__main__':
    main()
