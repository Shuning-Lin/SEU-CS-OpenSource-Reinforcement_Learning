import itertools
import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
import core
from logx import EpochLogger, DEFAULT_DATA_DIR


class PreferenceDataset:
    """
    The offline preference dataset D = {(x, y_w, y_l)} built by make_prefs.py.

    For every prefix x there are K = 4 completions scored by the ground-truth reward;
    each of the K*(K-1)/2 = 6 pairs becomes one preference, y_w being the completion
    with the higher reward. Unlike PPOBuffer in 1.ppo, which is refilled with fresh
    experience every epoch, this data is fixed: DPO never samples from the policy.
    """

    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError(f'Preference data not found: {path}. Run make_prefs.py first or set --pref_path.')
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.prompts = data['prompts']                     # N prefixes (lists of token ids)
        self.completions = data['completions']             # (N, K, T) token ids
        rewards = data['rewards'].numpy()                  # (N, K) ground-truth rewards
        pairs = []
        for n in range(len(self.prompts)):
            for j, k in itertools.combinations(range(rewards.shape[1]), 2):
                if rewards[n, j] > rewards[n, k]:
                    pairs.append((n, j, k))                # (prefix, preferred y_w, dispreferred y_l)
                elif rewards[n, k] > rewards[n, j]:
                    pairs.append((n, k, j))                # (a tie gives no preference, so no pair)
        if not pairs:
            raise ValueError('No preference pairs: use at least two completions per prompt with unequal rewards.')
        self.pairs = np.array(pairs, dtype=np.int64)
        self.T = self.completions.shape[-1]

    def __len__(self):
        return len(self.pairs)

    def get(self, idx):
        """The preference pairs with indices idx, as (prompts x, preferred y_w, dispreferred y_l)."""
        n, w, l = torch.as_tensor(self.pairs[idx]).T
        prompts = [self.prompts[i] for i in n.tolist()]
        return prompts, self.completions[n, w], self.completions[n, l]



def dpo(sft_path, pref_path, beta=0.1, lr=1e-6, batch_size=64, micro_batch_size=16, warmup_steps=150,
        epochs=1, eval_freq=100, num_test_prompts=256, seed=0, logger_kwargs=dict(), eval_batch_size=32):
    r"""
    Direct Preference Optimization (Rafailov et al. 2023)

    No reward model, no sampling during training, no critic. DPO fits the policy to
    the preferences with a simple binary cross-entropy loss (Eq. 7 of the paper):

        L_DPO = -E_{(x, y_w, y_l) ~ D} [ log sigmoid( beta * log pi(y_w|x) / pi_ref(y_w|x)
                                                    - beta * log pi(y_l|x) / pi_ref(y_l|x) ) ]

    and still optimizes the same KL-constrained objective as RLHF with PPO (Eq. 3):

        max_pi  E[ r(x, y) ] - beta * KL( pi(y|x) || pi_ref(y|x) )

    Args:
        sft_path (str): Folder of the SFT model. It is both the initial policy and
            the frozen reference policy pi_ref.

        pref_path (str): The preference data made by make_prefs.py.

        beta (float): Strength of the KL constraint (the paper: 0.05, 0.1, 1, 5).
            Roughly: a smaller beta lets the policy move further away from pi_ref,
            which gives a higher reward but also a higher KL. Typically denoted
            by :math:`\beta`.

        lr (float): Learning rate for the RMSprop optimizer.

        batch_size (int): Number of preference pairs per gradient step (the paper: 64).

        micro_batch_size (int): Number of pairs processed at once. The gradient of a
            batch is accumulated over batch_size / micro_batch_size micro-batches,
            which saves GPU memory and gives exactly the same gradient.

        warmup_steps (int): The learning rate grows linearly from 0 to lr over the
            first warmup_steps gradient steps (the paper: 150). Set 0 to disable warmup.

        epochs (int): Number of passes over the preference pairs.

        eval_freq (int): Every eval_freq gradient steps, evaluate the policy (reward
            and KL on the test prompts) and write one row of progress.csv
            (the paper: every 100 steps).

        num_test_prompts (int): Number of IMDb test prefixes used in the evaluation.

        eval_batch_size (int): Batch size for evaluation generation, KL and sentiment scoring.

        seed (int): Seed for random number generators.

        logger_kwargs (dict): Keyword args for EpochLogger.

    """

    for name, value in dict(beta=beta, lr=lr, batch_size=batch_size, micro_batch_size=micro_batch_size,
                            epochs=epochs, eval_freq=eval_freq, num_test_prompts=num_test_prompts,
                            eval_batch_size=eval_batch_size).items():
        if value <= 0:
            raise ValueError(f'{name} must be greater than 0 (got {value}).')
    if warmup_steps < 0:
        raise ValueError('warmup_steps must be 0 or greater.')

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config({k: v for k, v in locals().items() if k != 'logger'})

    # Random seed
    core.set_seed(seed)

    # Load the offline preference data
    data = PreferenceDataset(pref_path)
    T = data.T

    # Create the policy pi_theta (trainable) and the reference policy pi_ref (frozen), both start as pi_SFT
    policy = core.load_lm(sft_path)
    ref = core.load_lm(sft_path, trainable=False)
    logger.log('\nNumber of parameters: \t %d, \t preference pairs: %d\n' % (core.count_vars(policy), len(data)))

    # Only for evaluation: DPO itself never sees this reward function, it only sees the preferences
    tokenizer = core.load_tokenizer()
    reward_fn = core.SentimentReward()
    test_prompts = core.make_prompts('test', num_test_prompts, tokenizer)

    # Set up function for computing the DPO loss
    def compute_loss_dpo(prompts, chosen, rejected):
        B = len(prompts)

        # One forward pass for both completions: rows 0..B-1 hold y_w, rows B..2B-1 hold y_l
        input_ids, attention_mask = core.make_batch(prompts + prompts, torch.cat([chosen, rejected]))
        pi_logps = core.response_logprobs(policy, input_ids, attention_mask, T).sum(-1)       # log pi_theta(y|x), shape (2B,)
        with torch.no_grad():
            ref_logps = core.response_logprobs(ref, input_ids, attention_mask, T).sum(-1)     # log pi_ref(y|x),   shape (2B,)

        # DPO loss: the next lines follow the PyTorch code in Appendix B of the paper
        pi_yw_logps, pi_yl_logps = pi_logps[:B], pi_logps[B:]
        ref_yw_logps, ref_yl_logps = ref_logps[:B], ref_logps[B:]
        pi_logratios = pi_yw_logps - pi_yl_logps
        ref_logratios = ref_yw_logps - ref_yl_logps
        losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))

        # Useful extra info: the implicit reward r_hat(x, y) = beta * log pi_theta(y|x) / pi_ref(y|x)
        # (Section 5.1, "Your Language Model Is Secretly a Reward Model")
        chosen_rewards = beta * (pi_yw_logps - ref_yw_logps).detach()
        rejected_rewards = beta * (pi_yl_logps - ref_yl_logps).detach()
        info = dict(LossDPO=losses.mean().item(),
                    RewardAcc=(chosen_rewards > rejected_rewards).float().mean().item(),
                    RewardMargin=(chosen_rewards - rejected_rewards).mean().item(),
                    ChosenReward=chosen_rewards.mean().item(),
                    RejectedReward=rejected_rewards.mean().item())

        return losses.mean(), info

    # Set up optimizer (RMSprop, as in the paper) with a linear learning-rate warmup
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup_steps)))

    # Set up model saving
    logger.setup_pytorch_saver(policy)

    def update(prompts, chosen, rejected):
        optimizer.zero_grad()
        # Gradient accumulation: the scaled losses of the micro-batches add up to the mean loss of the batch
        for i in range(0, len(prompts), micro_batch_size):
            mb = slice(i, i + micro_batch_size)
            loss, info = compute_loss_dpo(prompts[mb], chosen[mb], rejected[mb])
            (loss * len(prompts[mb]) / len(prompts)).backward()
            # Weight metrics by pair count, including the final partial micro-batch.
            logger.store(**{key: np.full(len(prompts[mb]), value) for key, value in info.items()})
        optimizer.step()
        scheduler.step()

    def evaluate_and_log(epoch, step):
        reward, kl, texts, rewards = core.evaluate(policy, ref, reward_fn, tokenizer, test_prompts, T, batch_size=eval_batch_size)
        logger.log_tabular('Epoch', epoch)                        # pass over the preference data
        logger.log_tabular('Step', step)                          # number of gradient steps so far
        logger.log_tabular('Reward', reward)                      # average ground-truth reward on the test prompts (y-axis of Figure 2, left)
        logger.log_tabular('KL', kl)                              # average sequence-level KL(pi || pi_ref)       (x-axis of Figure 2, left)
        logger.log_tabular('LossDPO', average_only=True)          # starts at log(2) = 0.693, because pi_theta = pi_ref gives sigmoid(0) = 0.5
        logger.log_tabular('RewardAcc', average_only=True)        # how often the implicit reward ranks y_w above y_l
        logger.log_tabular('RewardMargin', average_only=True)     # average r_hat(x, y_w) - r_hat(x, y_l)
        logger.log_tabular('ChosenReward', average_only=True)     # average r_hat(x, y_w)
        logger.log_tabular('RejectedReward', average_only=True)   # average r_hat(x, y_l), usually falls faster than r_hat(x, y_w) rises
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
        for text, r in list(zip(texts, rewards))[:2]:
            logger.log('reward %.3f | %s' % (r, text.replace('\n', ' ')), color='cyan')
        return texts, rewards

    # Step 0 is pi_SFT itself (pi_theta = pi_ref, so KL = 0); there are no training statistics yet
    start_time = time.time()
    logger.store(LossDPO=np.nan, RewardAcc=np.nan, RewardMargin=np.nan, ChosenReward=np.nan, RejectedReward=np.nan)
    texts, rewards = evaluate_and_log(0, 0)

    # Main loop: every epoch is one pass over the shuffled preference pairs
    step = 0
    for epoch in range(epochs):
        perm = np.random.permutation(len(data))
        for i in range(0, len(data), batch_size):
            update(*data.get(perm[i:i + batch_size]))
            step += 1
            if step % eval_freq == 0:
                texts, rewards = evaluate_and_log(epoch, step)
    if step % eval_freq != 0:
        texts, rewards = evaluate_and_log(epoch, step)           # so that the last row describes the saved policy

    # Save the final policy, and its continuations of the test prompts
    logger.save_state()
    with open(osp.join(logger.output_dir, 'samples.txt'), 'w', encoding='utf-8') as f:
        for text, r in zip(texts, rewards):
            f.write('reward %.3f | %s\n' % (r, text.replace('\n', ' ')))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft_path', type=str, default=osp.join(DEFAULT_DATA_DIR, 'sft', 'sft_s0', 'pyt_save', 'model'))
    parser.add_argument('--pref_path', type=str, default=osp.join(DEFAULT_DATA_DIR, 'prefs', 'prefs.pt'))
    parser.add_argument('--beta', type=float, default=0.1)          # the paper sweeps beta in {0.05, 0.1, 1, 5}
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--micro_batch_size', type=int, default=16) # lower it if you run out of GPU memory
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--num_test_prompts', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--warmup_steps', type=int, default=150, help='Set 0 to disable learning-rate warmup.')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default=None)       # default: dpo_beta<beta>, e.g. dpo_beta0.1
    args = parser.parse_args()

    from logx import setup_logger_kwargs
    exp_name = args.exp_name or 'dpo_beta%g' % args.beta
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    dpo(args.sft_path, args.pref_path, beta=args.beta, lr=args.lr, batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size, epochs=args.epochs, eval_freq=args.eval_freq,
        seed=args.seed, logger_kwargs=logger_kwargs, warmup_steps=args.warmup_steps,
        num_test_prompts=args.num_test_prompts, eval_batch_size=args.eval_batch_size)
