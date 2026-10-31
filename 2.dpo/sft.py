import numpy as np
import torch
from torch.optim import Adam
import time
import core
from logx import EpochLogger


def sft(base_model=core.BASE_MODEL, num_reviews=25000, max_len=128, batch_size=32, lr=5e-5,    
        epochs=1, log_freq=100, max_new_tokens=48, seed=0, logger_kwargs=dict()):
    # Attention, if you want to amend parameter, you should do this at the "default" paramenter in parser.argument()
    """
    Supervised Fine-Tuning (SFT), stage 1 of the pipeline.

    A pretrained GPT-2 writes all kinds of internet text. We fine-tune it on IMDb
    movie reviews with plain maximum likelihood, so that it writes movie reviews.
    The result, pi_SFT, is used three times later:
        1) make_prefs.py samples the preference data from it,
        2) it is the frozen reference policy pi_ref of DPO and PPO-GT(maybe I don't add this, but who care?),
        3) it is the initial policy of DPO and PPO-GT.

    Args:
        base_model (str): Pretrained model to start from.

        num_reviews (int): Number of IMDb training reviews to fine-tune on
            (the paper: "a subset of the IMDB data").

        max_len (int): Each review is cut to its first max_len tokens. Our prompts
            are review beginnings, so the beginnings are what matter most.

        batch_size (int): Number of reviews per gradient step.

        lr (float): Learning rate for the Adam optimizer.

        epochs (int): Number of passes over the reviews (the paper: 1 epoch).

        log_freq (int): Write one row of progress.csv every log_freq gradient steps.

        max_new_tokens (int): Length of the sample continuations printed at the end.

        seed (int): Seed for random number generators.

        logger_kwargs (dict): Keyword args for EpochLogger.
    """

    # Reject invalid small-run settings before creating outputs or downloading models.
    for name, value in dict(num_reviews=num_reviews, batch_size=batch_size, lr=lr,
                            epochs=epochs, log_freq=log_freq, max_new_tokens=max_new_tokens).items():
        if value <= 0:
            raise ValueError(f'{name} must be greater than 0 (got {value}).')
    if max_len < 2:
        raise ValueError('max_len must be at least 2 for next-token prediction.')

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config({k: v for k, v in locals().items() if k != 'logger'})

    # Random seed(and the default seed is 0)
    core.set_seed(seed)         

    # Training data: the beginnings of IMDb training reviews, as lists of token ids
    tokenizer = core.load_tokenizer()
    texts = core.load_imdb_texts('train')
    idx = np.random.permutation(len(texts))[:num_reviews]
    reviews = [tokenizer.encode(texts[i][:2000])[:max_len] for i in idx]   # 2000 characters hold more than 128 tokens

    reviews = [review for review in reviews if len(review) >= 2]
    if not reviews:
        raise ValueError('No reviews with at least two tokens; check the data and num_reviews.')

    # Create the language model
    model = core.load_lm(base_model)
    logger.log('\nNumber of parameters: \t %d, \t training reviews: %d\n' % (core.count_vars(model), len(reviews)))

    # Set up function for computing the SFT loss (the usual language-modeling loss)
    def compute_loss_sft(batch):
        input_ids, attention_mask = core.make_batch(batch)                       # one review per row (left-padded)
        logits = core.lm_logits(model, input_ids, attention_mask)[:, :-1]        # the logits at position i predict token i+1
        logp = torch.log_softmax(logits, dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).float()          # both the context and the target are real text
        return -(logp * mask).sum() / mask.sum()                                 # maximize log pi(review), token by token
                                                                                 # oh, I want to emphasize teacher forcing
    # Set up optimizer and model saving
    optimizer = Adam(model.parameters(), lr=lr)
    logger.setup_pytorch_saver(model)

    def log_progress(epoch, step):
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('Step', step)
        logger.log_tabular('LossSFT', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    # Main loop: plain supervised learning
    start_time = time.time()
    step = 0
    for epoch in range(epochs):
        perm = np.random.permutation(len(reviews))
        for i in range(0, len(reviews), batch_size):
            batch = [reviews[j] for j in perm[i:i + batch_size]]

            optimizer.zero_grad()
            loss = compute_loss_sft(batch)
            loss.backward()
            optimizer.step()
            
            logger.store(LossSFT=loss.item())
            step += 1

            if step % log_freq == 0:
                log_progress(epoch, step)
    if step % log_freq != 0:
        log_progress(epoch, step)  # Even a one-step smoke run gets a progress row.

    # Save the model: data/sft/sft_s0/pyt_save/model is pi_SFT
    logger.save_state()

    # Have a look: the model should now write (something like) movie reviews
    prompts = core.make_prompts('test', 5, tokenizer, seed=seed)     # after surpervised fine tuning, sample from five film comments and let model contiue to write.
    completions = core.sample(model, prompts, max_new_tokens)
    for text in core.decode(tokenizer, prompts, completions):
        logger.log('Sample: ' + text.replace('\n', ' '), color='cyan')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default=core.BASE_MODEL)    # appoint base model, for example, GPT-2
    parser.add_argument('--num_reviews', type=int, default=25000)             # the number of review
    parser.add_argument('--max_len', type=int, default=128)                   
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--lr', type=float, default=5e-5)         
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=48)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='sft')
    args = parser.parse_args()

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed) 

    sft(base_model=args.base_model, num_reviews=args.num_reviews, max_len=args.max_len,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, seed=args.seed,
        log_freq=args.log_freq, max_new_tokens=args.max_new_tokens, logger_kwargs=logger_kwargs)
