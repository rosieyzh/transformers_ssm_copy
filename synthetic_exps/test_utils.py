import torch
from data_utils import get_eval_dataset
import numpy as np
import wandb
import math

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_score(args,tokenizer,x,pred,i):
    x_out = tokenizer.decode(x[i])
    pred_out = tokenizer.decode(pred[i])

    index = x_out.index('>')
    end_index = x_out.index('*')
    gt = x_out[index + 1: end_index]

    end_pred_idx = index + len(gt)
    pred_model = pred_out[index:end_pred_idx]

    print("gt:", gt)
    print("pred_model:", pred_model)

    str_correct = 1
    num_chars_correct = 0
    for tok_1, tok_2 in zip(gt, pred_model):
        if tok_1 != tok_2:
            str_correct = 0
        else:
            num_chars_correct += 1
    
    total_chars = len(gt)
    return str_correct, num_chars_correct, total_chars



def evaluation(args, model, tokenizer, TO_TOKEN):
    

    lengths = np.arange(args.min_eval_len, args.max_eval_len + 1, args.eval_len_interval)
    
    str_acc_mean_list = []
    char_accuracy_list = []
    print("\n")
    
    for ood_length in lengths:
        long_dataset = get_eval_dataset(args, tokenizer, TO_TOKEN, target_min_len=ood_length,target_max_len=ood_length)    
        if args.eval_task == "count":
            args.eval_num_batches = int(math.ceil(len(long_dataset.examples) / args.eval_batch_size))
            print(f"There are {long_dataset.examples} samples, with {args.eval_num_batches} to run.")

        str_correct_batch = np.zeros(args.eval_num_batches)
        len_each_batch = np.zeros(args.eval_num_batches)
        char_correct_batch = np.zeros(args.eval_num_batches)
        char_len_each_batch = np.zeros(args.eval_num_batches)

        for jj in range(args.eval_num_batches): 
            batch = next(iter(long_dataset))
            
            # print("-"*100)
            # print(f"EXAMPLE {batch['input'][0]}")
            # print("-"*100)
            # print(batch['input_ids'][-1][batch['mask'][-1]==1], batch['input_ids'][-1], batch['input'][-1])
            # print("*"*100)
            
            x = batch['input_ids'].to('cuda')
            
            with torch.no_grad():

                ##prediction
                if args.model=="lstm":
                    state = model.init_hidden(args.eval_batch_size, 'cuda')
                    logits, state = model(x, state)
                elif args.model=="mamba":
                    logits = model(x)[0]
                else:
                    logits = model(x)['logits']
                
                ##greedy decoding
                pred = torch.argmax(logits, dim=-1)


                ##evaluation
                for i in range(len(x)):
                    str_correct, num_chars_correct, total_chars = get_score(args,tokenizer,x,pred,i) 
                    str_correct_batch[jj] += str_correct
                    char_correct_batch[jj] += num_chars_correct
                    char_len_each_batch[jj] += total_chars
            
            len_each_batch[jj] = len(batch["input"])
            # print(f"current batch: {str_correct_batch[jj]} correct strings, {char_correct_batch[jj]} correct characters, {char_len_each_batch[jj]} number of characters")

        mean_str_acc = np.sum(str_correct_batch) / np.sum(len_each_batch)
        mean_char_acc = np.sum(char_correct_batch) / np.sum(char_len_each_batch)

        str_acc_mean_list.append(mean_str_acc)
        char_accuracy_list.append(mean_char_acc)
        
        if args.wandb:
            wandb.log({f"{args.eval_task}_mean_str_acc": mean_str_acc, f"{args.eval_task}_mean_char_acc": mean_char_acc, "Steps": ood_length})

        print(f"{args.eval_task}; len {ood_length}: {mean_str_acc}; char: {mean_char_acc}")
    print("\n")        
    return str_acc_mean_list, char_accuracy_list



