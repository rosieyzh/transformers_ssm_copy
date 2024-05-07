import numpy as np
import torch
import string
import torch.nn.functional as F
import random 

class NumberTokenizer:
    def __init__(self, TO_TOKEN, TO_CHAR):
        
        self.TO_TOKEN = TO_TOKEN
        self.TO_CHAR = TO_CHAR

        self.bos_token_id = TO_TOKEN['$']
        self.eos_token_id = TO_TOKEN['.']

    def __call__(self, x):
        encoded = [self.TO_TOKEN[c] for c in x]
        return torch.tensor(encoded, dtype=torch.int64)

    def decode(self, x):
        x = x.detach().cpu().numpy()
        decoded = [str(t) if t not in self.TO_CHAR else self.TO_CHAR[t] for t in x]
        return decoded

    def __len__(self):
        return len(self.TO_TOKEN)


def arr_to_str(x):
    return ''.join([str(n) for n in x])

def rand_num(length,vocab_size):
        string_ascii_lowercase = string.ascii_lowercase[:vocab_size]
        num = "".join(np.random.choice(list(string_ascii_lowercase),size=length))
        return arr_to_str(num)

def most_frequent_character(chars):
    char_count = {}
    for char in chars:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # Find the character(s) with the highest frequency
    max_freq = max(char_count.values())
    most_frequent_chars = [char for char, freq in char_count.items() if freq == max_freq]
    return most_frequent_chars


def generate_str_unique_ngram(len1, n_gram, length_answer,vocab_size):

    counter_max_ngram = 10
    counter_ngram = 0
    unique = False
    while not unique:
       num1 = rand_num(len1,vocab_size)
       max_limit = len(num1) - n_gram - length_answer -1 if length_answer > 0 else len(num1) - n_gram -1
       list_ngrams = [num1[idx : idx + n_gram] for idx in range(max_limit)]
       unique_n_grams = []
       for ng in list_ngrams:
         if list_ngrams.count(ng) == 1:
           unique_n_grams.append(ng)
       if unique_n_grams:
         unique = True
       counter_ngram +=1
       if counter_ngram >= counter_max_ngram:
          raise ValueError(f"Unable to find a unique {n_gram}-gram in a string of length {len1}!")
    return num1, list_ngrams  


def sample_str(len1,task,vocab_size=26, len2=None):
    example_list = ['$']
    if task == "count":
        # len1 gives length of output tokens
        lower_number = np.random.randint(0, vocab_size - len1 + 1)
        higher_number = lower_number + len1 - 1
        example_list += [str(lower_number), str(higher_number) , '>']
        example_list += [str(x) for x in range(lower_number, higher_number + 1)]
    elif task == "mode":
        # select five tokens at random
        tokens = np.random.choice(vocab_size, 5)
        sequence = [str(random.choice(tokens)) for _ in range(len1)]
        most_freq = most_frequent_character(sequence)
        if len(most_freq) == 1:
            example_list += sequence + ['>', most_freq[0]]
        else:
            to_replace = random.sample(most_freq, k=2)
            replace_indices = [i for i, x in enumerate(sequence) if x == to_replace[0]]
            index = random.choice(replace_indices)
            sequence[index] = to_replace[1] # this is now the most frequent index
            example_list += sequence + ['>', str(to_replace[1])]
    elif task == "copy":
        # Sample a sequence without replacement
        sequence = [str(x) for x in random.sample(list(range(vocab_size)), k=len1)]
        example_list += sequence + ['>'] + sequence
    elif task == "copy_repeat":
        # Sample a sequence with replacement
        sequence = [str(x) for x in random.choices(list(range(vocab_size)), k=len1)]
        example_list += sequence + ['>'] + sequence
    elif task == "sort":
        sequence = random.sample(list(range(vocab_size)), k=len1)
        example_list += [str(x) for x in sequence] + ['>']
        sequence.sort()
        example_list += [str(x) for x in sequence]
    elif task == "addition":
        num_1 = [str(random.randint(0, 9)) for _ in range(len1)]
        num_2 = [str(random.randint(0, 9)) for _ in range(len2)]
        str_num_1 = ''.join(num_1)
        str_num_2 = ''.join(num_2)
        result = int(str_num_1) + int(str_num_2)
        if len2 > len1:
            seq_1 = ['0'] * (len2 - len1 + 1) + [c for c in num_1]
            seq_2 = ['0'] + [c for c in num_2]
        else:
            seq_1 = ['0'] + [c for c in num_1]
            seq_2 = ['0'] * (len1 - len2 + 1) + [c for c in num_2]
        result = [c for c in str(result)]
        if len(seq_2) > len(result):
            result.insert(0, '0')
        example_list += seq_1 + ['+'] + seq_2 + ['>'] + result
    elif task == "parity":
        sequence = np.random.choice(vocab_size, size=len1, replace=True)
        parity = np.sum(sequence) % 2
        example_list += [str(x) for x in sequence] + ['>', str(parity)]
    elif task == "and":
        sequence = np.random.choice(vocab_size, size=len1, replace=True)
        if np.all((sequence == 1)):
            example_list += [str(x) for x in sequence] + ['>', '1']
        else:
            example_list += [str(x) for x in sequence] + ['>', '0']

    example_list += ['.']

    return example_list

class CopyDataset:
    def __init__(self, tokenizer, vocab_size=26, n_gram=0, length_answer=-1, train_task="copy", sequence_length=220, min_length=20, max_length=50, num_examples=1000, batch_size=8): 
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.train_task = train_task
        self.vocab_size = vocab_size
        self.n_gram = n_gram
        self.length_answer = length_answer

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'input_ids': [], 'mask': []}
        
        minimal_required_length = self.n_gram if self.n_gram > 0 else 0
        minimal_required_length += self.length_answer if self.length_answer > 0 else 0
        if self.min_length <= minimal_required_length:
            raise ValueError(f"Minimum length is set to {self.min_length} and is smaller than the required one {minimal_required_length}")
        
        minimal_required_length = 3 # start token, > token, end token
        if self.train_task in ["copy", "copy_repeat", "sort"]:
          minimal_required_length += 2 * self.max_length
        elif self.train_task == "count":
          minimal_required_length += 2 + self.max_length
        elif self.train_task in ["mode", "parity", "and"]:
          minimal_required_length += 1 + self.max_length
        elif self.train_task == "addition":
          minimal_required_length += 1 + 3 * (self.max_length + 1)

        if self.sequence_length <= minimal_required_length:
            raise ValueError(f"Strings of size {self.max_length} do not fit in a context of size {self.sequence_length} for task {self.train_task}. Increase your context length !")

        for _ in range(self.batch_size):
            prospective_len = 0
            full_list = []
            example_mask = []
            while prospective_len < self.sequence_length:
              
                ##sample a string  
                len1 = np.random.randint(self.min_length, self.max_length+1)
                len2 = None
                if self.train_task == "addition":
                    len2 = np.random.randint(self.min_length, self.max_length+1)
                example_list = sample_str(len1, self.train_task, self.vocab_size, len2)
              
                ###setting up mask for training loss
                if self.train_task == "count":
                    example_mask_tmp = [0] * (4) + [1] * (len(example_list) - 4)
                elif self.train_task in ["copy", "copy_repeat", "sort", "parity", "and", "mode"]:
                    example_mask_tmp = [0] * (len1+2) + [1] * (len(example_list) - len1-2)
                elif self.train_task == "addition":
                    example_mask_tmp = [0] * (5 + 2 * max(len1, len2)) + [1] * (len(example_list) - (5 + 2 * max(len1, len2)))

                #packing the context with examples
                if prospective_len+len(example_list) > self.sequence_length:
                    remaining_len = self.sequence_length - prospective_len
                    remaining_mask_len = self.sequence_length - prospective_len
                    full_list += example_list[:remaining_len]
                    example_mask += [0]*(remaining_mask_len)
                    break
                else:
                    full_list += example_list
                    prospective_len += len(example_list)
                    example_mask += example_mask_tmp

            assert len(full_list) == len(example_mask)
            example_ids = self.tokenizer(full_list)
            example_mask = torch.tensor(example_mask)

            batch['input'].append(full_list)
            batch['input_ids'].append(example_ids)
            batch['mask'].append(example_mask)

        batch['input_ids'] = torch.stack(batch['input_ids'], dim=0)
        batch['mask'] = torch.stack(batch['mask'], dim=0)
        return batch




class EvalCopyDataset:
    def __init__(self, tokenizer, TO_TOKEN, vocab_size=26, n_gram=3, length_answer=-1, eval_task="copy", sequence_length=220, min_length=8, max_length=30, num_examples=1000, batch_size=8):
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.TO_TOKEN = TO_TOKEN
        self.vocab_size = vocab_size
        self.eval_task = eval_task
        self.n_gram = n_gram
        self.length_answer = length_answer
        if self.eval_task == "count": # instead of randomly sampling batches, we enumerate all possibilities instead
            assert min_length == max_length
            self.examples = [(i, i + self.max_length - 1) for i in range(0, self.vocab_size - self.max_length + 1)]
            self.cur_index = 0

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'input_ids': [], 'mask': []}
        minimal_required_length = 3 # start token, > token, end token
        if self.eval_task in ["copy", "copy_repeat", "sort"]:
          minimal_required_length += 2 * self.max_length
        elif self.eval_task == "count":
          minimal_required_length += 2 + self.max_length
        elif self.eval_task in ["mode", "parity", "and"]:
          minimal_required_length += 1 + self.max_length
        elif self.eval_task == "addition":
          minimal_required_length += 1 + 3 * (self.max_length + 1)

        if self.sequence_length <= minimal_required_length:
            raise ValueError(f"Strings of size {self.max_length} do not fit in a context of size {self.sequence_length} for task {self.eval_task}. Increase your context length !")

        for _ in range(self.batch_size):
            if self.eval_task != "count":
                ##sample a string  
                len1 = np.random.randint(self.min_length, self.max_length+1)
                len2 = None
                if self.eval_task == "addition":
                    len2 = np.random.randint(self.min_length, self.max_length+1)
                example_list = sample_str(len1, self.eval_task, self.vocab_size, len2)
            else:
                if self.cur_index == len(self.examples): break
                lower_number, higher_number = self.examples[self.cur_index]
                example_list = ["$"]
                example_list += [str(lower_number), str(higher_number) , '>']
                example_list += [str(x) for x in range(lower_number, higher_number + 1)]
                self.cur_index += 1
            
            ##fill the context with padding
            example_ids = self.tokenizer(example_list)
            if len(example_ids) < self.sequence_length:
                example_ids = F.pad(example_ids, (0, self.sequence_length-len(example_ids)), value=self.TO_TOKEN['*'])
            

            ###setting up mask for training loss
            if self.eval_task == "count":
                example_mask = [0] * (4) + [1] * (len(example_list) - 4) + [0] * (self.sequence_length-len(example_list))
            elif self.eval_task in ["copy", "copy_repeat", "sort", "parity", "and", "mode"]:
                example_mask = [0] * (len1+2) + [1] * (len(example_list) - len1-2) + [0] * (self.sequence_length-len(example_list))
            elif self.eval_task == "addition":
                example_mask = [0] * (5 + 2 * max(len1, len2)) + [1] * (len(example_list) - (5 + 2 * max(len1, len2))) + [0] * (self.sequence_length-len(example_list))
            
            assert len(example_ids)==len(example_mask)
            example_mask = torch.tensor(example_mask)
            batch['input'].append(example_list)
            batch['input_ids'].append(example_ids)
            batch['mask'].append(example_mask)
        batch['input_ids'] = torch.stack(batch['input_ids'], dim=0)
        batch['mask'] = torch.stack(batch['mask'], dim=0)
        return batch


def get_tokenizer(args):
    tokens = [str(x) for x in range(args.vocab_size)]
    letters = dict(zip(tokens, range(args.vocab_size)))

    symbols = {'$': len(letters), '>': len(letters)+1, '.': len(letters)+2, '*': len(letters)+3}

    TO_TOKEN = {**letters, **symbols}

    if args.train_task == "addition":
        TO_TOKEN = {**TO_TOKEN, **{'+': len(TO_TOKEN)}}

    TO_CHAR = {v:k for k,v in TO_TOKEN.items()}

    tokenizer = NumberTokenizer(TO_TOKEN, TO_CHAR)
    return tokenizer, TO_TOKEN, TO_CHAR


def get_train_dataset(args,tokenizer):

    train_dataset = CopyDataset(
            tokenizer,
            vocab_size=args.vocab_size,
            n_gram=args.n_gram,
            length_answer=args.length_answer,
            train_task=args.train_task,
            sequence_length=args.context_len,
            min_length=args.min_train_len,
            max_length=args.max_train_len,
            batch_size=args.train_batch_size,
            )
    
    return train_dataset


def get_eval_dataset(args, tokenizer, TO_TOKEN, target_min_len,target_max_len):
    
    eval_dataset = EvalCopyDataset(
            tokenizer, 
            TO_TOKEN, 
            vocab_size=args.vocab_size,
            n_gram=args.n_gram,
            length_answer=args.length_answer,
            eval_task=args.eval_task,
            sequence_length=args.context_len,
            min_length=target_min_len, 
            max_length=target_max_len,
            batch_size=args.eval_batch_size,
            )

    return eval_dataset
