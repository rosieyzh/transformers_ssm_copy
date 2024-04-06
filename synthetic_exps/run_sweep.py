import os
import argparse
import itertools

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int)
    parser.add_argument("--job_id", type=int)
    parser.add_argument("--num_inits", type=int)
    parser.add_argument("--num_data", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    combination = list(itertools.product(list(range(args.num_inits)), list(range(args.num_data))))
    init_seed, data_seed = combination[args.task_id - 1]

    # os.system(
    #     f'python3 main.py --init_seed {init_seed} --data_seed {data_seed}  --model "T_rope" --hidden_size=512 --layers=6 --heads=8 --train_task "copy" --eval_task  "copy" --lr 3e-4 --scheduler "cosine" --warmup 500 --steps 50000 --train_batch_size 32 --min_train_len 1 --max_train_len 20 --eval_batch_size 32 --eval_num_batches 100 --min_eval_len 20 --max_eval_len 101 --eval_len_interval 10 --context_len 512 --eval_context_len 512 --wandb_name {args.job_id}_{args.task_id} --wandb_group 19m'
    # )
    os.system(
        f'python3 main.py --init_seed {init_seed} --data_seed {data_seed}  --model "T_rope" --hidden_size=768 --layers=12 --heads=12 --train_task "copy" --eval_task  "copy" --lr 3e-4 --scheduler "cosine" --warmup 500 --steps 50000 --train_batch_size 32 --min_train_len 1 --max_train_len 20 --eval_batch_size 32 --eval_num_batches 100 --min_eval_len 20 --max_eval_len 101 --eval_len_interval 10 --context_len 512 --eval_context_len 512 --wandb_name {args.job_id}_{args.task_id} --wandb_group 19m'
    )
