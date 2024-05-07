import itertools
import os
import wandb
import json
import argparse
from copy import copy
import random
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import re

from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, AutoConfig

from tqdm import tqdm
from collections import Counter 
from pathlib import Path

import string
from model_utils import get_model
from data_utils import get_train_dataset, get_tokenizer, get_eval_dataset
from train_utils import train, save_model
from test_utils import evaluation

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_all(seed: int):
    """Seed all rng objects."""

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    ##seed
    parser.add_argument('--init_seed', default=0, type=int, help="seed for initializing model")
    parser.add_argument('--data_seed', default=0, type=int, help="seed for setting data ordering")

    ##task
    parser.add_argument('--train_task',choices=["count", "mode", "copy", "copy_repeat", "sort", "addition", "parity", "and"],
                        required=True, help="tasks to train the model")
    parser.add_argument('--eval_task',choices=["count", "mode", "copy", "copy_repeat", "sort", "addition", "parity", "and"],
                        required=True, help="tasks to evaluate the model")
    
    parser.add_argument('--vocab_size', default=26, type=int, help="vocabulary size in the strings. for some tasks, this will be set by default.")
    
    parser.add_argument('--n_gram', default=0, type=int, 
            help='''length of the n-gram when training/evaluating on 'prefix_ngram', 'suffix_ngram', 'duplicate_ngram'.
            Set 0 for 'copy' task.''')
    parser.add_argument('--length_answer', default=0, type=int, 
            help="length of the answer to be returned. Set 0 if no constraint on the length of the answer.")

    #model
    parser.add_argument('--model', choices=['T_nope', 'T_rope', 'T_alibi', "T_hard_alibi",  'lstm', 'mamba'],
            required=True, help='''models starting by 'T' are transformers with different positional embeddings. Other choices
            are mamba and lstm.''')
    parser.add_argument('--hidden_size', default=1024, type=int, help="Hidden size of the models")
    parser.add_argument('--layers', default=12, type=int, help="Number of layers in the models.")
    parser.add_argument('--heads', default=16, type=int, help="Number of heads in the transformer models.")
    parser.add_argument('--num_masked_heads', default=8, type=int, help='''Only when model = ''T_hard_alibi''. 
            Number of heads where we apply hard alibi. The remaining heads are set to nope.''')
    parser.add_argument('--state_dim', default=32, type=int, help='''Only when model = ''mamba''. 
            Sets the state dimension of the model.''')

    #optimization
    parser.add_argument('--scheduler', default="linear", type=str, help="choice of scheduler")
    parser.add_argument('--lr', default=1e-5, type=float, help="choice of learning rate")
    parser.add_argument('--wd', default=0.1, type=float, help="weight decay")
    parser.add_argument('--grad_clip', default=1.0, type=float, help="gradient clipping")
    parser.add_argument('--warmup', default=500, type=int, help="number of warmup steps")
    parser.add_argument('--epochs', default=1, type=int, help="number of epochs")
    parser.add_argument('--steps', default=2000, type=int, help="number of steps for each epoch")
    

    parser.add_argument('--train_batch_size', default=8, type=int, help="training batch size")
    parser.add_argument('--eval_batch_size', default=8, type=int, help="evaluation batch size")
    parser.add_argument('--eval_num_batches', default=3, type=int, help='''number of batches to use for evaluation.
            useful to have a mean + std over results.''')
    
    parser.add_argument('--min_train_len', default=5, type=int, help="minimum length of a training example")
    parser.add_argument('--max_train_len', default=20, type=int, help="maximum length of a training example")
    parser.add_argument('--min_eval_len', default=10, type=int, help="minimum length of an evaluation example")
    parser.add_argument('--max_eval_len', default=20, type=int, help="maximum length of an evaluation example")
    parser.add_argument('--eval_len_interval', default=1, type=int, help="interval of eval length")
    

    ##context length
    parser.add_argument('--context_len', default=220, type=int, help="context length during training")
    parser.add_argument('--eval_context_len', default=220, type=int, help="context length at evaluation time")

    ##wandb
    parser.add_argument('--wandb', default=True, type=bool, help="use wandb to log experiments")
    parser.add_argument('--wandb_project', default='ood_random_variation', type=str, help="wandb project name")
    parser.add_argument('--wandb_group', type=str, help="wandb group name")
    parser.add_argument('--wandb_name', type=str, help="wandb run name")
    
    return parser.parse_args()






args = parse_args()

print(args)




## Get tokenizer

if args.train_task in ["copy_repeat", "parity", "and"]: # Ensure vocab size is correct for Boolean tasks
        assert args.vocab_size == 2, f"Vocab size for Boolean tasks and copying with repeat tokens should be 2, but is {args.vocab_size}."
elif args.train_task in ["addition"]:
        assert args.vocab_size == 10, f"Vocab size for addition should be 2, but is {args.vocab_size}."
tokenizer, TO_TOKEN, TO_CHAR = get_tokenizer(args)

## Get model
seed_all(args.init_seed)
model = get_model(args, tokenizer)

print("^"*100)
print(model)
print(f"Number of parameters of the model: {count_parameters(model)}")
print("^"*100)

## Get train dataset
seed_all(args.data_seed)
train_dataset = get_train_dataset(args,tokenizer) 




batch = next(iter(train_dataset))

print("-"*100)
print(f"EXAMPLE {batch['input'][0]}")
print("-"*100)
print(batch['input_ids'][-1][batch['mask'][-1]==1], batch['input_ids'][-1], batch['input'][-1])
print("*"*100)


## init wandb
if args.wandb:
        name = args.wandb_name if args.wandb_name else f'init_{args.init_seed}_data_{args.data_seed}'
        group = args.wandb_group if args.wandb_group else 'default'
        wandb_dir = "results/"
        wandb.init(
                dir=wandb_dir,
                project=args.wandb_project,
                group=group,
                name=name,
                config=args,
        )
        wandb.define_metric('Steps')
        wandb.define_metric("*", step_metric="Steps")

## train the model
train(args,model,train_dataset,TO_TOKEN)



## save model
save_model(args, model)


## evaluation of the model

print("###EVALUATION")

model.eval()

str_acc_mean_list, char_accuracy_list = evaluation(args, model,tokenizer,TO_TOKEN)


print(args)

print("DONE")



