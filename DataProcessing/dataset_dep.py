import os
import glob
import logging

import torch
import torch.nn.functional as F

import os
import sys
import pickle
from DataProcessing.structure import load_ct,ct2dot_and_pairs
from DataProcessing.binary_tree import get_tree

import re
import pandas as pd
import numpy as np

from functools import partial
import torch
from torch.utils.data import Dataset

from utils.utils import logger



class Example(object):
    def __init__(self,
                 p_id=None,
                 seq=None,
                 arc=None,
                 rel=None ):
        self.p_id = p_id
        self.seq = seq
        self.arc = arc
        self.rel = rel


def save(filepath, obj, message=None):
    if message is not None:
        logging.info("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

def pickle_load_large_file(filepath):
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj

def load(filepath):
    return pickle_load_large_file(filepath)


def read_examples(args,file_path, session):
    examples_file = args.cache_data +session+ ".pkl"
    
    relation_dic = args.relation_dic

    if not os.path.exists(examples_file):
        data_dict = process_rna_files(file_path+'/ct_seq')
        examples = []
        for p_id, key in enumerate(data_dict.keys(),start=1):

            # print(key)
            # print(data_dict[key])
            seq = data_dict[key]['seq']
            seq_uppercase = convert_to_upper(seq) #ACGU big letter
        
            

            # generate arc and rel
            normal_pairs = data_dict[key]['normal_pairs']
            pseudo_pairs = data_dict[key]['pseudo_pairs']

            normal_structure = data_dict[key]['normal_structure'] #...((((....))))
            pair_list, parse_pair, pse_pair = get_tree(normal_structure,pseudo_pairs) 

            parse_pair = [(x+1, y+1) for x, y in parse_pair]
            pse_pair = [(x+1, y+1) for x, y in pse_pair]
            pair_list = [(x+1, y+1) for x, y in pair_list]

            # 初始化arc和rel矩阵
            arc = np.zeros(len(seq_uppercase) + 1, dtype=int)
            rel = np.zeros(len(seq_uppercase) + 1, dtype=int)

            # 构建heads字典
            heads = {word: head for head, word in parse_pair}
            heads.update({word: head for head, word in pair_list})
            heads.update({word: head for head, word in pse_pair})

            pseudo_flat_list = [item for pair in pseudo_pairs for item in pair]

            # 构建relations字典
            relations = {}
            for head, word in parse_pair:
                if (head, word) in normal_pairs or (word, head) in normal_pairs:
                    relations[word] = 'stem'
                else:
                    relations[word] = 'stemnect'
            for head, word in pse_pair:
                if (head, word) in pseudo_flat_list or (word, head) in pseudo_flat_list:
                    relations[word] = 'pseudo'
                else:
                    relations[word] = 'stemnect'
            for head, word in pair_list:
                relations[word] = 'loop'

            # 遍历seq_uppercase，填充arc和rel
            for i in range(1, len(seq_uppercase) + 1):
                if i in heads:
                    head = heads[i]
                    relation = relations[i]
                else:
                    head = 0
                    relation = 'root'
                arc[i] = head
                rel[i] = relation_dic[relation]
        
            examples.append(
            Example(
                        p_id=p_id,  # 1
                        seq=seq_uppercase,  # 'GGGCGCGGCGCCGGCC
                        arc=arc,  #[0,2,4,4,0] 
                        rel=rel  #[0,1,3,4,2] 5 class

                    ))
        save(examples_file, examples)

    else:
        logger.info('loading cache_data {}'.format(examples_file))
        examples = load(examples_file)
        logger.info('examples size is {}'.format(len(examples)))

    return examples


def convert_to_upper(seq):
    # Helper function to convert lowercase letters to uppercase in the sequence
    return ''.join(c.upper() if c.islower() else c for c in seq)

def process_rna_files(dir_name):
    # 用于存储结果的字典
    result_dict = {}

    # 遍历目录下的所有文件
    for filename in os.listdir(dir_name):
        # print(filename)
        # 去除文件扩展名
        file_name_without_ext = os.path.splitext(filename)[0]
        # 检查文件是否是.ct文件
        if filename.endswith(".ct"):      
            # 加载ct
            ct = load_ct(os.path.join(dir_name, filename))
            # 转换ct为dot

            normal_structure, normal_pairs, pseudo_pairs = ct2dot_and_pairs(ct[1], len(ct[0]))
            if pseudo_pairs:
                print(f"伪结存在于文件: {filename}")

            # 把结果存储到字典中
            result_dict[file_name_without_ext] = {"normal_structure": normal_structure, "normal_pairs": normal_pairs, "pseudo_pairs": pseudo_pairs}    

    for filename in os.listdir(dir_name):
        # 去除文件扩展名
        file_name_without_ext = os.path.splitext(filename)[0]
        # 检查文件是否是.seq文件
        if filename.endswith(".seq"):
            # 读取文件的最后一行
            with open(os.path.join(dir_name, filename), 'r') as f:
                seq = f.readlines()[-1].strip()
            # 如果最后一个字符是数字，删除它
            if seq[-1].isdigit():
                seq = seq[:-1]
            # 如果字典中已经有对应的键，那么把seq添加到那个键的值中
            if file_name_without_ext in result_dict:
                result_dict[file_name_without_ext]["seq"] = seq
                # 检查structure和seq的长度是否一致
                if len(seq) != len(result_dict[file_name_without_ext]["normal_structure"]):
                    print(f"Warning: Length of seq and structure are not consistent for {file_name_without_ext}")
            # 如果字典中还没有对应的键，那么新建一个键并设置它的值为seq
            else:
                print(f"File {file_name_without_ext} does not have a corresponding .ct file")

    return result_dict

# from utils.finetuning_argparse import get_argparse
# args = get_argparse().parse_args()
# args.cache_data = "./data/mhs"
# file_path = '/home/ke/Documents/RNA/mxfold2-data/data/bpRNA_dataset-canonicals/TR0'
# examples =  read_examples(args,file_path, 'test')


class Biaffine_Dataset(Dataset):
    def __init__(self, args, examples, data_type):
        self.tokenizer = args.tokenizer
        # self.max_len = args.max_len
        self.q_ids = list(range(len(examples)))
        self.examples = examples
        self.is_train = True if data_type == 'train' else False

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, index):
        return self.q_ids[index], self.examples[index]

    def _create_collate_fn(self):
        def collate(examples):

            # p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)

            batch_token_ids = []
            batch_seq = []
            batch_tag = []
            batch_arc = []
            batch_rel = []


            for example in examples:
                seq = example[1].seq # 'GGGCGCGGCGCCGGCC' no cls and seq
                batch_seq.append(seq)
                arc = example[1].arc
                batch_arc.append(arc)
                rel = example[1].rel
                batch_rel.append(rel)
            
            batch_token_ids,mask = generate_token(self.tokenizer,batch_seq)
            max_seq_len = batch_token_ids.size()[1]  # 最大序列长度

            batch_arc_tensor = [torch.tensor(arc, dtype=torch.long) for arc in batch_arc]
            batch_rel_tensor = [torch.tensor(rel, dtype=torch.long) for rel in batch_rel]

            # 对每个tensor进行填充
            batch_arc_padded = torch.stack([F.pad(arc, (0, max_seq_len - arc.size(0)), "constant", 0) for arc in batch_arc_tensor])
            batch_rel_padded = torch.stack([F.pad(rel, (0, max_seq_len - rel.size(0)), "constant", 0) for rel in batch_rel_tensor])

            return batch_seq, batch_token_ids, mask, batch_arc_padded, batch_rel_padded
        
        return partial(collate)



def generate_token(alphabet, seq_strs):
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len
            + int(alphabet.prepend_bos)
            + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    
    # 初始化mask为全0
    mask = torch.zeros_like(tokens, dtype=torch.int64)

    for i, seq_str in enumerate(seq_strs):
        seq_len = len(seq_str)
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
            mask[i, 0] = 1  # 标记为非填充位置
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[i, int(alphabet.prepend_bos): seq_len + int(alphabet.prepend_bos)] = seq
        mask[i, int(alphabet.prepend_bos): seq_len + int(alphabet.prepend_bos)] = 1  # 更新mask以标记非填充位置
        if alphabet.append_eos:
            tokens[i, seq_len + int(alphabet.prepend_bos)] = alphabet.eos_idx
            mask[i, seq_len + int(alphabet.prepend_bos)] = 1  # 标记为非填充位置
    
    return tokens, mask