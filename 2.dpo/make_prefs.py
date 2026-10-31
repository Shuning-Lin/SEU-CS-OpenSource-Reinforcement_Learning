import os
import os.path as osp
import time
import numpy as np
import torch
import core
from logx import DEFAULT_DATA_DIR


def make_prefs(sft_path, num_prompts=25000, num_samples=4, max_new_tokens=48, chunk_size=64,
               seed=0, out_dir=osp.join(DEFAULT_DATA_DIR, 'prefs')):
    """
    Stage 2 of the pipeline: build the offline preference dataset (Appendix C.1).

    "We use [the SFT] model to sample 4 completions for 25000 prefixes and create
    6 preference pairs for each prefix using the ground-truth reward model."

    In the real world a human compares two completions and says which one is better.
    Here the sentiment classifier plays the human: between two completions of the same
    prefix, the one with the higher p(positive) is the preferred one (y_w), the other
    one is the dispreferred one (y_l). With 4 completions there are 4*3/2 = 6 pairs.

    We save the 4 completions and their rewards; dpo.py turns them into the 6 pairs.

    Args:
        sft_path (str): Folder of the SFT model (the policy that writes the completions).

        num_prompts (int): Number of prefixes x, taken from the IMDb train split.

        num_samples (int): Number of completions sampled for each prefix.

        max_new_tokens (int): Length T of every completion (in tokens).

        chunk_size (int): Number of prefixes processed at once.

        seed (int): Seed for random number generators.

        out_dir (str): Where to save prefs.pt and a human-readable preview.txt.
    """
    for name, value in dict(num_prompts=num_prompts, max_new_tokens=max_new_tokens,
                            chunk_size=chunk_size).items():
        if value <= 0:
            raise ValueError(f'{name} must be greater than 0 (got {value}).')
    if num_samples < 2:
        raise ValueError('num_samples must be at least 2 to create preference pairs.')

    core.set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = core.load_tokenizer()
    policy = core.load_lm(sft_path, trainable=False)
    reward_fn = core.SentimentReward()
    prompts = core.make_prompts('train', num_prompts, tokenizer, seed=seed)

    if not prompts:
        raise ValueError('No prompts available; check the dataset and num_prompts.')
    
    N, K, T = len(prompts), num_samples, max_new_tokens
    completions = torch.zeros((N, K, T), dtype=torch.long)
    rewards = torch.zeros((N, K))

    start_time = time.time()
    for i in range(0, N, chunk_size):
        chunk = prompts[i:i + chunk_size]                            # attention here has two "for"
        repeated = [p for p in chunk for _ in range(K)]              # every prefix K times: x1 x1 x1 x1 x2 x2 ...
        c = core.sample(policy, repeated, T, batch_size=len(repeated))
        r = reward_fn(core.decode(tokenizer, repeated, c))
        completions[i:i + len(chunk)] = c.view(len(chunk), K, T)
        rewards[i:i + len(chunk)] = torch.as_tensor(r).view(len(chunk), K)
        if (i // chunk_size) % 50 == 0:
            done = i + len(chunk)
            print('%d / %d prefixes, %.0fs elapsed, about %.0fs left'
                  % (done, N, time.time() - start_time, (time.time() - start_time) / done * (N - done)), flush=True)

    torch.save(dict(prompts=prompts, completions=completions, rewards=rewards), osp.join(out_dir, 'prefs.pt'))

    # Some statistics about the preference pairs
    r = rewards.numpy()
    gaps = [abs(r[n, j] - r[n, k]) for n in range(N) for j in range(K) for k in range(j + 1, K)]
    print('Saved %d prefixes x %d completions to %s' % (N, K, osp.join(out_dir, 'prefs.pt')))
    print('Average reward of the SFT completions: %.3f' % rewards.mean())
    print('Reward gap |r(y_w) - r(y_l)| of the pairs: mean %.3f, fraction of pairs with gap > 0.5: %.3f'
          % (np.mean(gaps), np.mean(np.array(gaps) > 0.5)))

    # A human-readable preview: look at it to understand what DPO learns from!
    with open(osp.join(out_dir, 'preview.txt'), 'w', encoding='utf-8') as f:
        for n in range(min(20, N)):
            f.write('=' * 80 + '\nPREFIX: %s\n' % tokenizer.decode(prompts[n]))
            for k in np.argsort(-rewards[n].numpy()):
                f.write('  reward %.3f | %s\n' % (rewards[n, k], tokenizer.decode(completions[n, k]).replace('\n', ' ')))
    print('Wrote a preview of the data to %s' % osp.join(out_dir, 'preview.txt'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_path', type=str, default=osp.join(DEFAULT_DATA_DIR, 'sft', 'sft_s0', 'pyt_save', 'model'))      # the model which has been trained, and use it to generate continuation
    parser.add_argument('--num_prompts', type=int, default=25000)                                                              # the prompts from IMDb         
    parser.add_argument('--num_samples', type=int, default=4)                                                                  # how many continuation for each prompt(at least 2)
    parser.add_argument('--len', type=int, default=48)                                                                         # the number of token for each continuation
    parser.add_argument('--chunk_size', type=int, default=64, help='Prefixes per generation chunk; lower to save GPU memory.') # set the number of prompts for each batch, controling the occupation of vedio memory
    parser.add_argument('--out_dir', type=str, default=osp.join(DEFAULT_DATA_DIR, 'prefs'))                                    # the direction for preference data output
    parser.add_argument('--seed', '-s', type=int, default=0)                                                                   # random seed
    args = parser.parse_args()

    make_prefs(args.sft_path, num_prompts=args.num_prompts, num_samples=args.num_samples,
               max_new_tokens=args.len, chunk_size=args.chunk_size, seed=args.seed, out_dir=args.out_dir)
