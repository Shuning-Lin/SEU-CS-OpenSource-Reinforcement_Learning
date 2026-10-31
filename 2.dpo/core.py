"""
Core building blocks shared by every script of the DPO vs PPO-GT sentiment experiment
(Rafailov et al. 2023, "Direct Preference Optimization", Section 6.1 and Appendix C.1).

Compared with 1.ppo/core.py, the "environment" is now text generation:
    state   s_t : the prompt x plus the t tokens generated so far
    action  a_t : the next token, i.e. a Discrete action space with 50257 actions
    policy      : a GPT-2 language model, pi(a_t | s_t) = Categorical(logits=GPT2(s_t))
    episode     : one continuation y = (a_0, ..., a_{T-1}) of exactly T tokens
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')  # huggingface.co is blocked in mainland China, so download through a mirror (delete this line if you don't need it)

import random
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')                # Allow faster float32 matmuls on supported GPUs; speed/accuracy depend on hardware.
BASE_MODEL = 'gpt2'                                       # 124M parameters. The paper uses 'gpt2-large' (774M), see README.md
REWARD_MODEL = 'siebert/sentiment-roberta-large-english'  # the ground-truth reward model used in the paper
EOS_ID = 50256                                            # id of '<|endoftext|>' in every GPT-2 tokenizer


def count_vars(module):
    # Calculate the number of the parameters of a net
    return sum([int(np.prod(p.shape)) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def set_seed(seed):
    # Tip: Python, Numpy and Pytorch all have their own random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#==================== Data: IMDb movie reviews ====================

def load_imdb_texts(split):
    """All the reviews of one IMDb split ('train' or 'test'), with the html line breaks removed."""
    dataset = load_dataset('stanfordnlp/imdb', split=split)
    return [text.replace('<br />', ' ') for text in dataset['text']]


def make_prompts(split, num_prompts, tokenizer, seed=0, min_len=2, max_len=8):
    """
    The prompts are prefixes of IMDb reviews of length 2-8 tokens (Appendix C.1), e.g.
        "I rented I AM CURIOUS"   ->   the policy has to continue the review.
    The same (split, num_prompts, seed) always gives the same prompts, in every script.
    Returns a list of prompts, each one a list of token ids.
    """
    texts = load_imdb_texts(split)
    rng = np.random.RandomState(seed)                                  # a private generator, independent of set_seed
    idx = rng.permutation(len(texts))[:num_prompts]
    lens = rng.randint(min_len, max_len + 1, size=len(idx))
    return [tokenizer.encode(texts[i][:200])[:n] for i, n in zip(idx, lens)]  # 200 characters hold far more than 8 tokens


#==================== The policy: a GPT-2 language model ====================

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no padding token; padded positions are masked out anyway
    return tokenizer


def load_lm(path, trainable=True):           # the variable "trainable" decides whether to froze the model
    """
    Load a GPT-2 language model: a pretrained one (e.g. 'gpt2') or one saved by our
    scripts (e.g. 'data/sft/sft_s0/pyt_save/model').
    """
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    # Attention: eval() only switches dropout off, gradients still work. We never use dropout, otherwise the
    # same policy would give different log-probs for the same text.
    model.eval()                             # the function here is to close Dropout, which can set some activation value to zero
    if not trainable:
        model.requires_grad_(False)          # e.g. the frozen reference policy pi_ref
    return model


def make_batch(prompts, completions=None):
    """
    Put a list of prompts (and optionally a (B, T) tensor of completions) into one tensor.
    The prompts are padded on the LEFT, so that every completion occupies the same last T columns:
        [pad pad  x1  x2 | y1 ... yT]
        [pad  x1  x2  x3 | y1 ... yT]
    Returns input_ids and attention_mask (0 marks padding), both of shape (B, P+T).
    """
    B, P = len(prompts), max(len(p) for p in prompts)
    T = 0 if completions is None else completions.shape[1]
    input_ids = torch.full((B, P + T), EOS_ID, dtype=torch.long)
    attention_mask = torch.zeros((B, P + T), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, P - len(p):P] = torch.as_tensor(p)
        attention_mask[i, P - len(p):] = 1
    if completions is not None:
        input_ids[:, P:] = completions
    return input_ids.to(DEVICE), attention_mask.to(DEVICE)


def lm_logits(model, input_ids, attention_mask):
    """One forward pass of the policy: the logits (B, L, vocab) of the next-token distribution at every position."""
    # Critical: the position ids must skip the left padding. Without them the model treats the padding as
    # part of the text, and the log-probs differ from the ones used while generating (by several nats!).

    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)   # you can understand "cumsum" as cumulative sum from left to right

    logits = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).logits

    # Ban the end-of-text token, so every continuation has exactly T tokens (a fixed-horizon episode).
    # The most negative float is used instead of -inf, so the KL computations never meet "inf - inf".
    logits[..., EOS_ID] = torch.finfo(logits.dtype).min           # yeah, I am little confused about this code, but let it go!
    return logits


def response_logprobs(model, input_ids, attention_mask, T, with_entropy=False):
    """
    log pi(y_t | x, y_<t) of the last T tokens (the completion) of every row, shape (B, T).
    Summing over t gives log pi(y | x), the quantity DPO works with.
    """
    logits = lm_logits(model, input_ids, attention_mask)[:, -T-1:-1]   # the logits at position i predict token i+1
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, input_ids[:, -T:].unsqueeze(-1)).squeeze(-1)
    if with_entropy:
        return logp, -(logp_all.exp() * logp_all).sum(-1)              # entropy of every next-token distribution
    return logp


@torch.no_grad()
def sample(model, prompts, T, batch_size=128):
    """
    Let the policy write one continuation of exactly T tokens for every prompt. Like ac.step() in 1.ppo,
    each token is sampled from Categorical(logits): temperature 1 and no top-k / top-p tricks, because
    policy gradients and KL measurements assume that we sample from pi itself.
    Returns a LongTensor of shape (len(prompts), T) on the CPU.
    """
    completions = []
    for i in range(0, len(prompts), batch_size):
        input_ids, attention_mask = make_batch(prompts[i:i + batch_size])
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                             max_new_tokens=T, suppress_tokens=[EOS_ID], pad_token_id=EOS_ID)
        completions.append(out[:, -T:].cpu())   # generate() returns prompt + completion; keep the completion
    return torch.cat(completions)


def decode(tokenizer, prompts, completions):
    """Turn token ids back into text (prompt + completion), e.g. for the reward model."""
    return [tokenizer.decode(list(p) + c.tolist()) for p, c in zip(prompts, completions)]


#==================== The ground-truth reward: a sentiment classifier ====================

class SentimentReward:
    """
    The ground-truth reward r*(x, y) = p(positive | x, y) of a pretrained sentiment classifier.
    The paper uses it to (1) decide which of two completions is preferred and (2) evaluate every policy.
    PPO-GT, the "oracle" baseline, also uses it directly as its reward.
    """

    def __init__(self, name=REWARD_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()    # .eval() is to close dropout
        self.positive = self.model.config.label2id['POSITIVE']   # {'NEGATIVE': 0, 'POSITIVE': 1}

    @torch.no_grad()
    def __call__(self, texts, batch_size=64):
        probs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            probs.append(torch.softmax(self.model(**enc).logits, dim=-1)[:, self.positive].cpu())
        return torch.cat(probs).numpy()


@torch.no_grad()
def evaluate(policy, ref, reward_fn, tokenizer, prompts, T, batch_size=32, seed=1234):
    """
    The evaluation behind Figure 2 (left) of the paper. Sample one continuation per test prompt, then compute
        reward : the average ground-truth reward p(positive | x, y)
        kl     : the average sequence-level KL(pi || pi_ref), i.e. the sum over the T timesteps of the
                 KL divergence between the two next-token distributions (footnote 3 of the paper)
    """
    # Use the same random numbers in every evaluation (without disturbing the training random numbers),
    # so that two evaluations differ because the policy changed, not because of luck.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        completions = sample(policy, prompts, T, batch_size=batch_size)

    kls = []
    for i in range(0, len(prompts), batch_size):
        input_ids, attention_mask = make_batch(prompts[i:i + batch_size], completions[i:i + batch_size])
        logp_pi = torch.log_softmax(lm_logits(policy, input_ids, attention_mask)[:, -T-1:-1], dim=-1)  # (B, T, vocab)
        logp_ref = torch.log_softmax(lm_logits(ref, input_ids, attention_mask)[:, -T-1:-1], dim=-1)
        kl_t = (logp_pi.exp() * (logp_pi - logp_ref)).sum(-1)   # (B, T): KL(pi || pi_ref) at every timestep
        kls.append(kl_t.sum(-1).cpu())                           # (B,):   summed over the T timesteps

    texts = decode(tokenizer, prompts, completions)
    rewards = reward_fn(texts, batch_size=batch_size)
    return float(rewards.mean()), float(torch.cat(kls).mean()), texts, rewards


#==================== The critic of PPO-GT ====================

class LMCritic(nn.Module):
    """
    The value function V(s_t) of PPO-GT, analogous to MLPCritic in 1.ppo/core.py:
    a GPT-2 body (initialized from the SFT model) plus a linear layer that outputs one number per position.
    """

    def __init__(self, path):
        super().__init__()
        self.body = AutoModel.from_pretrained(path)                    # GPT-2 without its language-model head
        self.head = nn.Linear(self.body.config.hidden_size, 1)
        self.to(DEVICE)
        self.eval()                                                    # no dropout, see load_lm

    def forward(self, input_ids, attention_mask, T):
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)   # skip the left padding, see lm_logits
        h = self.body(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
        # V(s_t) of the T states in which a completion token is chosen (the same positions as in response_logprobs)
        return self.head(h[:, -T-1:-1]).squeeze(-1)                    # (B, T). Critical to ensure v has right shape.
