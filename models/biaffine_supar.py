# -*- coding: utf-8 -*-
"""
@Time : 3/Jun/2024

@DESCRIPTION: based on supar biaffine dependency parser
only use the biaffine model

@Author : ke

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MLP, Biaffine
from utils.utils import MIN
import RNA
from typing import List
# from torch.distributions.distribution import Distribution
# from torch.distributions.utils import lazy_property
import torch.autograd as autograd
from typing import Iterable, Union
from modules.tree import DependencyCRF, MatrixTree
from modules.pretrained import TransformerEmbedding
from modules.focalloss import FocalLoss
from modules.pretrained import RNAfmEmbedding


class BiaffineModel(nn.Module):

    def __init__(self,args,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=0.33,
                 scale=0,
                 ):
        super().__init__()

        self.args = args

        if args.is_pse:
            n_rels = 6
        else:
            n_rels = 5

        if self.args.embedding == 'one-hot':
            self.encoder = F.one_hot()
            self.bert_hidden = 5    

        elif self.args.embedding == 'RNA-fm':
            self.encoder = self.args.encoder.requires_grad_(self.args.finetune)
            self.encoder_dropout = nn.Dropout(p=0.1, inplace=False)
            self.bert_hidden = 640

        elif self.args.embedding == 'roberta-base':
            self.encoder = self.args.encoder(name=self.args.embedding, n_layers=4,pooling = 'mean',pad_index=1, finetune=self.args.finetune)
            self.encoder_dropout = nn.Dropout(p=0.1, inplace=False)
            self.bert_hidden = self.encoder.n_out
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding}")


        self.arc_mlp_d = MLP(n_in=self.bert_hidden, n_out=n_arc_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_arc_mlp]  [B, L, 500]
        self.arc_mlp_h = MLP(n_in=self.bert_hidden, n_out=n_arc_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_arc_mlp]  [B, L, 500]

        self.rel_mlp_d = MLP(n_in=self.bert_hidden, n_out=n_rel_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_rel_mlp]  [B, L, 100]
        self.rel_mlp_h = MLP(n_in=self.bert_hidden, n_out=n_rel_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_rel_mlp]  [B, L, 100]

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)

        if self.args.loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.loss == 'focal_loss':
            self.criterion = FocalLoss(gamma=2, alpha=None, size_average=True)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss}")


    def forward(self,
                batch_token_ids,
                ):

        if self.args.embedding == 'one-hot':
            bert_output = self.encoder(batch_token_ids, num_classes=5).float()

        elif self.args.embedding == 'RNA-fm':
            x = self.encoder(batch_token_ids, need_head_weights=False, repr_layers=[12], return_contacts=False)
            x = x["representations"][12]
            bert_output = self.encoder_dropout(x)   

        elif self.args.embedding == 'roberta-base':
            batch_token_ids = batch_token_ids.unsqueeze(-1)
            x = self.encoder(batch_token_ids)
            bert_output = self.encoder_dropout(x)  

        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding}") 

        seq_len = bert_output.size(1)

        arc_d = self.arc_mlp_d(bert_output) # [batch_size, seq_len, n_arc_mlp] [B, L, 500]
        arc_h = self.arc_mlp_h(bert_output)  # [B, L, 500]
        rel_d = self.rel_mlp_d(bert_output) # [B, L, 100]
        rel_h = self.rel_mlp_h(bert_output) # [B, L, 100]


    
        pad_index =self.args.tokenizer.padding_idx
        mask = batch_token_ids.ne(pad_index) if len(batch_token_ids.shape) < 3 else batch_token_ids.ne(pad_index).any(-1) #set pad to be false, but the start and end is true

        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)  #MIN = -1e32 set the pad to be MIN

        # mask_bool = mask.bool()  # Convert mask to Boolean
        # s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(mask_bool.unsqueeze(1), MIN) # Use the Boolean mask

        # s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN) # [batch_size, seq_len, seq_len] [B, L, L]

        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1) # [batch_size, seq_len, seq_len, n_rels] [B, L, L, 5]

        return s_arc, s_rel
    

    def loss(self, s_arc, s_rel, arcs, rels, mask):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.


        Returns:
            ~torch.Tensor:
                The training loss.
        """

        mask = mask.bool()

        s_arc, arcs = s_arc[mask], arcs[mask] # [batch_size, seq_len, seq_len]  [B, L, L]
        s_rel, rels = s_rel[mask], rels[mask] # [batch_size, seq_len, seq_len, n_rels]  [B, L, L, 5]
        s_rel = s_rel[torch.arange(len(arcs)), arcs] # [batch_size, seq_len, n_rels]  [B, L, 5]

        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        loss = arc_loss + rel_loss

        return arc_loss, rel_loss, loss
    

    

    def decode(self, s_arc, s_rel, seed, beta,mask, tree, proj):
        # print(s_arc.shape)
        # 1. 生成新的 constrain matrix
        row_softmax = F.softmax(s_arc, dim=-1)
        col_softmax = F.softmax(s_arc, dim=-2)
        s_arc_softmax = row_softmax
            # 创建一个大于阈值的掩码
        # threshold_mask = (s_arc_softmax > 0.1).float()


        # 1.1 保留最后一个维度索引的最大三个值
        _, top_3_indices = torch.topk(s_arc_softmax, k=1, dim=-1)
        top_3_mask = torch.zeros_like(s_arc).scatter_(-1, top_3_indices, 1.0) 


        # torch.set_printoptions(edgeitems=mask.size(-1), sci_mode=False, precision=2,
        #        linewidth=1000)
        # print(s_arc)
        # print(top_3_mask)
        
        # 1.3 只保留上三角矩阵（不包括对角线）
        upper_triangular_mask = torch.triu(torch.ones_like(s_arc), diagonal=1)
        # print(upper_triangular_mask)

        # 1.2 应用原来的 mask

        masked_s_arc = s_arc_softmax * mask.unsqueeze(1).float()

        # 1.4 只保留 s_rel 中为 stem 和 pseudo 的
        label_map = self.args.relation_dic
        stem_index = label_map['stem']
    
        pseudo_index = label_map['pseudo']


        _, max_label_indices = torch.max(s_rel, dim=-1)

        stem_pseudo_mask = ((max_label_indices == stem_index) | (max_label_indices == pseudo_index))
        # print(stem_pseudo_mask.shape)
        
        # 组合所有约束条件
        constrain_matrix = top_3_mask*  upper_triangular_mask * stem_pseudo_mask.float()
        # print(constrain_matrix)
        # constrain_matrix = (max_label_indices == stem_index) 
        # 应用 constrain matrix 到 s_arc
        constrained_s_arc = masked_s_arc * constrain_matrix
        # print(constrained_s_arc)
        
        # 新增步骤：取每行和每列的最大值
        col_max = torch.argmax(constrained_s_arc, dim=-1)
        col_one = torch.zeros_like(constrained_s_arc).scatter(-1, col_max.unsqueeze(-1), 1.0)
        row_max = torch.argmax(constrained_s_arc, dim=-2)
        row_one = torch.zeros_like(constrained_s_arc).scatter(-2, row_max.unsqueeze(-2), 1.0)
        pred_contacts = row_one * col_one
     
        return pred_contacts
  
    def contact_map(self, s_arc, s_rel, mask, is_softmax):

        mask = mask.bool()   
        lens = mask.sum(1)

        if is_softmax:
            s_arc = F.softmax(s_arc, dim=-1)

        label_map = self.args.relation_dic
        stem_index = label_map['stem']
        pseudo_index = label_map['pseudo']

        _, max_label_indices = torch.max(s_rel, dim=-1)

        stem_pseudo_mask = ((max_label_indices == stem_index) | (max_label_indices == pseudo_index))

        contact_map = s_arc * stem_pseudo_mask.float() * mask

        return contact_map


def istree(sequence: List[int], proj: bool = False, multiroot: bool = False) -> bool:

    r"""
    Checks if the arcs form an valid dependency tree.

    Args:
        sequence (List[int]):
            A list of head indices.
        proj (bool):
            If ``True``, requires the tree to be projective. Default: ``False``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Returns:
        ``True`` if the arcs form an valid tree, ``False`` otherwise.

    Examples:
        >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
        True
        >>> CoNLL.istree([3, 0, 0, 3], proj=True)
        False
    """

    
    if proj and not isprojective(sequence):
        return False
    n_roots = sum(head == 0 for head in sequence)
    if n_roots == 0:
        return False
    if not multiroot and n_roots > 1:
        return False
    if any(i == head for i, head in enumerate(sequence, 1)):
        return False
    return next(tarjan(sequence), None) is None

def isprojective(sequence: List[int]) -> bool:
    # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/models/dep/biaffine/transform.py')
    # print('isprojective')
    r"""
    Checks if a dependency tree is projective.
    This also works for partial annotation.

    Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
    which are hard to detect in the scenario of partial annotation.

    Args:
        sequence (List[int]):
            A list of head indices.

    Returns:
        ``True`` if the tree is projective, ``False`` otherwise.

    Examples:
        >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
        False
        >>> CoNLL.isprojective([3, -1, 2])
        False
    """

    pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i + 1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return (False, f'Crossing arcs between tokens (h,d) = ({hi}, {di}) and (h,d) = ({hj}, {dj})')
            if lj <= hi <= rj and hj == di:
                return (False, f'Crossing arcs between tokens (h,d) = ({hi}, {di}) and (h,d) = ({hj}, {dj})')
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                return (False, f'Non-projective configuration between tokens (h,d) = ({hi}, {di}) and (h,d) = ({hj}, {dj})')
    return True, None

def tarjan(sequence) :

    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices making up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):

        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)




