# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'Haber-GPT-2-Small'
wandb_run_name='tinystories-nanogpt-v1.0-30M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 96

gradient_accumulation_steps = 1

# this makes total number of tokens be 300B
max_iters = 11200
lr_decay_iters = 11200

# eval stuff
eval_interval = 800
eval_iters = 85
log_interval = 100

# weight decay
weight_decay = 1e-1
warmup_iters = 20

decay_lr = False
learning_rate = 5e-4 # with baby networks can afford to go a bit higher
lr_decay_iters = 11200 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small


# baby GPT model :)
block_size = 512
vocab_size = 8192
n_layer = 8
n_head = 4
n_embd = 512
dropout = 0.1
bias = False

dataset = "tiny-stories-vocab-8k"
dtype = "float16"
compile = False
