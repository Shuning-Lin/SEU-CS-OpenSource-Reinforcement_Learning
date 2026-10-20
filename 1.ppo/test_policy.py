from pathlib import Path
import gymnasium as gym
import torch
import core  # noqa: F401  (the saved model is a pickled core.MLPActorCritic)
import numpy as np


def load_policy(fpath, deterministic=False, itr=None):
    """Load a numbered checkpoint, or the latest one by default."""
    save_dir = Path(fpath) / 'pyt_save'

    if itr is None:
        checkpoints = [
            p for p in save_dir.glob('model*.pt')
            if p.is_file() and p.stem[5:].isdigit()
        ]
        fname = max(
            checkpoints,
            key=lambda p: int(p.stem[5:]),
            default=save_dir / 'model.pt',
        )
    else:
        fname = save_dir / f'model{itr}.pt'

    if not fname.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {fname}')

    print(f'Loading from {fname}')
    model = torch.load(fname, map_location='cpu', weights_only=False)
    model.eval()

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            if deterministic:
                # most likely action (Discrete) / mean action (Box)
                pi = model.pi._distribution(x)
                return pi.mode.numpy()
            return model.act(x)

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100):

    rets, lens = [], []
    o, _ = env.reset()
    ep_ret, ep_len, n = 0, 0, 0
    while n < num_episodes:
        a = get_action(o)
        o, r, terminated, truncated, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if terminated or truncated or (ep_len == max_ep_len):
            rets.append(ep_ret)
            lens.append(ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0
            n += 1

    print('AverageEpRet %.3f 	 MinEpRet %.3f 	 MaxEpRet %.3f 	 AverageEpLen %.1f'
          %(np.mean(rets), np.min(rets), np.max(rets), np.mean(lens)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str, help='experiment folder, e.g. data/ppo/ppo_s0')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument(
        '--itr', type=int, default=None,
        help='checkpoint epoch; default: latest',
    )
    args = parser.parse_args()

    env = gym.make(args.env, render_mode=None if args.norender else 'human')
    get_action = load_policy(args.fpath, args.deterministic, args.itr)
    run_policy(env, get_action, args.len, args.episodes)
