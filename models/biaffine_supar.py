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
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property
import torch.autograd as autograd
from typing import Iterable, Union
from modules.tree import DependencyCRF, MatrixTree
from modules.pretrained import TransformerEmbedding
from modules.focalloss import FocalLoss
from modules.pretrained import RNAfmEmbedding

# class BiaffineModel(nn.Module):

#     def __init__(self,args,
#                  n_arc_mlp=500,
#                  n_rel_mlp=100,
#                  mlp_dropout=0.33,
#                  scale=0,
#                  ):
#         super().__init__()

#         self.args = args

#         if args.is_pse:
#             n_rels = 6
#         else:
#             n_rels = 5

#         if self.args.embedding == 'one-hot':
#             self.encoder = F.one_hot()
#             self.bert_hidden = 25    

#         elif self.args.embedding == 'RNA-fm':
#             self.encoder = self.args.encoder.requires_grad_(self.args.finetune)
#             self.encoder_dropout = nn.Dropout(p=0.33)
#             self.bert_hidden = 640

#         elif self.args.embedding == 'roberta-base':
#             self.encoder = self.args.encoder(name=self.embedding, n_layers=4,pooling = 'mean',pad_index=1, finetune=self.args.finetune)
#             self.encoder_dropout = nn.Dropout(p=0.33)
#             self.bert_hidden = self.encoder.n_out
#         else:
#             raise ValueError(f"Unsupported embedding type: {self.embedding}")

#         n_tag_mlp=256
        
#         self.mlp_tag1 = MLP(n_in=self.bert_hidden, n_out=n_arc_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_arc_mlp]  [B, L, 500]
#         self.mlp_tag2 = MLP(n_in=n_arc_mlp, n_out=n_tag_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_tag_mlp]  [B, L, 256]

#         self.arc_mlp_d1 = MLP(n_in=self.bert_hidden, n_out=n_arc_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_arc_mlp]  [B, L, 500]
#         self.arc_mlp_d2 = MLP(n_in=n_arc_mlp + n_tag_mlp, n_out=n_arc_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_arc_mlp]  [B, L, 500]     


#         self.arc_mlp_h1 = MLP(n_in=self.bert_hidden, n_out=n_arc_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_arc_mlp]  [B, L, 500]
#         self.arc_mlp_h2 = MLP(n_in=n_arc_mlp + n_tag_mlp, n_out=n_arc_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_arc_mlp]  [B, L, 500]


#         self.rel_mlp_d1 = MLP(n_in=self.bert_hidden, n_out=n_rel_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_rel_mlp]  [B, L, 100]
#         self.rel_mlp_d2 = MLP(n_in=n_rel_mlp + n_tag_mlp, n_out=n_rel_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_rel_mlp]  [B, L, 100]

#         self.rel_mlp_h1 = MLP(n_in=self.bert_hidden, n_out=n_rel_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_rel_mlp]  [B, L, 100]
#         self.rel_mlp_h2 = MLP(n_in=n_rel_mlp + n_tag_mlp, n_out=n_rel_mlp, dropout=mlp_dropout) # [batch_size, seq_len, n_rel_mlp]  [B, L, 100]

#         self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
#         self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)

#         if self.args.loss == 'cross_entropy':
#             self.criterion = nn.CrossEntropyLoss()
#         elif self.args.loss == 'focal_loss':
#             self.criterion = FocalLoss(gamma=2, alpha=None, size_average=True)
#         else:
#             raise ValueError(f"Unsupported loss type: {self.loss}")


#     def forward(self,
#                 batch_token_ids=None,
#                 ):

#         if self.args.embedding == 'one-hot':
#             bert_output = self.encoder(batch_token_ids, num_classes=25).float()

#         elif self.args.embedding == 'RNA-fm':
#             x = self.encoder(batch_token_ids, need_head_weights=False, repr_layers=[12], return_contacts=False)
#             x = x["representations"][12]
#             bert_output = self.encoder_dropout(x)   

#         elif self.args.embedding == 'roberta-base':
#             batch_token_ids = batch_token_ids.unsqueeze(-1)
#             x = self.encoder(batch_token_ids)
#             bert_output = self.encoder_dropout(x)  

#         else:
#             raise ValueError(f"Unsupported embedding type: {self.embedding}") 

#         seq_len = bert_output.size(1)

#         tag = self.mlp_tag1(bert_output)  # [B, L, 500]
#         tag_hidden = self.mlp_tag2(tag) # [B, L, 256]

#         arc_d = self.arc_mlp_d1(bert_output) # [batch_size, seq_len, n_arc_mlp] [B, L, 500]
#         arc_d = torch.cat([arc_d, tag_hidden], dim=-1)  # [B, L, 500+256]
#         arc_d = self.arc_mlp_d2(arc_d) # [B, L, 500]

#         arc_h = self.arc_mlp_h1(bert_output)  # [B, L, 500]
#         arc_h = torch.cat([arc_h, tag_hidden], dim=-1) # [B, L, 500+256]
#         arc_h = self.arc_mlp_h2(arc_h) # [B, L, 500]

#         rel_d = self.rel_mlp_d1(bert_output) # [B, L, 100]
#         rel_d = torch.cat([rel_d, tag_hidden], dim=-1) # [B, L, 100+256]
#         rel_d = self.rel_mlp_d2(rel_d) # [B, L, 100]

#         rel_h = self.rel_mlp_h1(bert_output) # [B, L, 100]
#         rel_h = torch.cat([rel_h, tag_hidden], dim=-1) # [B, L, 100+256]
#         rel_h = self.rel_mlp_h2(rel_h) # [B, L, 100]

    
#         pad_index =self.args.tokenizer.padding_idx
#         mask = batch_token_ids.ne(pad_index) if len(batch_token_ids.shape) < 3 else batch_token_ids.ne(pad_index).any(-1) #set pad to be false, but the start and end is true

#         s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)  #MIN = -1e32 set the pad to be MIN

#         # mask_bool = mask.bool()  # Convert mask to Boolean
#         # s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(mask_bool.unsqueeze(1), MIN) # Use the Boolean mask

#         # s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN) # [batch_size, seq_len, seq_len] [B, L, L]

#         s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1) # [batch_size, seq_len, seq_len, n_rels] [B, L, L, 5]

#         return s_arc, s_rel
    
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
            self.bert_hidden = 25    

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
            bert_output = self.encoder(batch_token_ids, num_classes=25).float()

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
    


    def calculate_free_energies(sequences, structures):
        # 确保输入的序列和结构列表长度相同
        if len(sequences) != len(structures):
            raise ValueError("Sequences and structures must have the same length.")
        
        # 存储每个序列和结构的自由能结果
        energies = []

        # 循环遍历每个序列和结构组合
        for sequence, structure in zip(sequences, structures):
            # 创建RNA折叠复合体
            fc = RNA.fold_compound(sequence)
            # 计算并存储该结构的自由能
            energy = fc.eval_structure(structure)
            energies.append(energy)

        return energies


    def decode(self, s_arc, s_rel, seed, beta,mask, tree, proj):
        
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        mask = mask.bool()   
        lens = mask.sum(1)

        s_arc_probs = F.softmax(s_arc, dim=-1)
        s_rel_probs = F.softmax(s_rel[:, :, :, 3], dim=-1)

        if beta != 0:
            s_arc = s_arc_probs + beta * s_rel_probs

        if seed == -1:
            arc_preds = s_arc.argmax(-1)
        else:
            torch.manual_seed(seed)

            # Initialize an empty list to store arc_preds for each batch
            batch_arc_preds = []

            # Loop through each batch
            for batch in s_arc_probs:
                # Apply multinomial sampling for the current batch
                # Since batch is 2D now ([133, 133]), this is valid
                arc_pred = torch.multinomial(batch, num_samples=1).squeeze()  # num_samples=1 for one sample per row
                batch_arc_preds.append(arc_pred)

            # Convert list of tensors back into a single tensor
            arc_preds = torch.stack(batch_arc_preds)

        bad = [not istree(seq[1:i+1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            arc_preds[bad] = (DependencyCRF if proj else MatrixTree)(s_arc[bad], mask[bad].sum(-1)).argmax
        
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds




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

    # pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
    # for i, (hi, di) in enumerate(pairs):
    #     for hj, dj in pairs[i + 1:]:
    #         (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
    #         if li <= hj <= ri and hi == dj:
    #             return False
    #         if lj <= hi <= rj and hj == di:
    #             return False
    #         if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
    #             return False
    # return True
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
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/fn.py')
        # print('connect')
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




