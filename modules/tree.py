# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple, Union, Iterable

import torch
import torch.nn as nn
from modules.dist import StructuredDistribution
# from supar.structs.fn import mst 
from modules.semiring import LogSemiring, Semiring
from modules.fn import diagonal_stripe, expanded_stripe, stripe, pad
from torch.distributions.utils import lazy_property

MIN = -1e32

def tarjan(sequence: Iterable[int]) -> Iterable[int]:
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
            
import sys
sys.setrecursionlimit(2000)  # You can adjust this limit as needed


def chuliu_edmonds(s: torch.Tensor) -> torch.Tensor:
    r"""
    ChuLiu/Edmonds algorithm for non-projective decoding :cite:`mcdonald-etal-2005-non`.

    Some code is borrowed from `tdozat's implementation`_.
    Descriptions of notations and formulas can be found in :cite:`mcdonald-etal-2005-non`.

    Notes:
        The algorithm does not guarantee to parse a single-root tree.

    Args:
        s (~torch.Tensor): ``[seq_len, seq_len]``.
            Scores of all dependent-head pairs.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[seq_len]`` for the resulting non-projective parse tree.

    .. _tdozat's implementation:
        https://github.com/tdozat/Parser-v3
    """

    s[0, 1:] = MIN
    # prevent self-loops
    s.diagonal()[1:].fill_(MIN)
    # select heads with highest scores
    tree = s.argmax(-1)
    # return the cycle finded by tarjan algorithm lazily
    cycle = next(tarjan(tree.tolist()[1:]), None)
    # if the tree has no cycles, then it is a MST
    if not cycle:
        return tree
    # indices of cycle in the original tree
    cycle = torch.tensor(cycle)
    # indices of noncycle in the original tree
    noncycle = torch.ones(len(s)).index_fill_(0, cycle, 0)
    noncycle = torch.where(noncycle.gt(0))[0]

    def contract(s):
        # heads of cycle in original tree
        cycle_heads = tree[cycle]
        # scores of cycle in original tree
        s_cycle = s[cycle, cycle_heads]

        # calculate the scores of cycle's potential dependents
        # s(c->x) = max(s(x'->x)), x in noncycle and x' in cycle
        s_dep = s[noncycle][:, cycle]
        # find the best cycle head for each noncycle dependent
        deps = s_dep.argmax(1)
        # calculate the scores of cycle's potential heads
        # s(x->c) = max(s(x'->x) - s(a(x')->x') + s(cycle)), x in noncycle and x' in cycle
        #                                                    a(v) is the predecessor of v in cycle
        #                                                    s(cycle) = sum(s(a(v)->v))
        s_head = s[cycle][:, noncycle] - s_cycle.view(-1, 1) + s_cycle.sum()
        # find the best noncycle head for each cycle dependent
        heads = s_head.argmax(0)

        contracted = torch.cat((noncycle, torch.tensor([-1])))
        # calculate the scores of contracted graph
        s = s[contracted][:, contracted]
        # set the contracted graph scores of cycle's potential dependents
        s[:-1, -1] = s_dep[range(len(deps)), deps]
        # set the contracted graph scores of cycle's potential heads
        s[-1, :-1] = s_head[heads, range(len(heads))]

        return s, heads, deps

    # keep track of the endpoints of the edges into and out of cycle for reconstruction later
    s, heads, deps = contract(s)

    # y is the contracted tree
    y = chuliu_edmonds(s)
    # exclude head of cycle from y
    y, cycle_head = y[:-1], y[-1]

    # fix the subtree with no heads coming from the cycle
    # len(y) denotes heads coming from the cycle
    subtree = y < len(y)
    # add the nodes to the new tree
    tree[noncycle[subtree]] = noncycle[y[subtree]]
    # fix the subtree with heads coming from the cycle
    subtree = ~subtree
    # add the nodes to the tree
    tree[noncycle[subtree]] = cycle[deps[subtree]]
    # fix the root of the cycle
    cycle_root = heads[cycle_head]
    # break the cycle and add the root of the cycle to the tree
    tree[cycle[cycle_root]] = noncycle[cycle_head]

    return tree

def mst(scores: torch.Tensor, mask: torch.BoolTensor, multiroot: bool = False) -> torch.Tensor:
    r"""
    MST algorithm for decoding non-projective trees.
    This is a wrapper for ChuLiu/Edmonds algorithm.

    The algorithm first runs ChuLiu/Edmonds to parse a tree and then have a check of multi-roots,
    If ``multiroot=True`` and there indeed exist multi-roots, the algorithm seeks to find
    best single-root trees by iterating all possible single-root trees parsed by ChuLiu/Edmonds.
    Otherwise the resulting trees are directly taken as the final outputs.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        multiroot (bool):
            Ensures to parse a single-root tree If ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting non-projective parse trees.

    Examples:
        >>> scores = torch.tensor([[[-11.9436, -13.1464,  -6.4789, -13.8917],
                                    [-60.6957, -60.2866, -48.6457, -63.8125],
                                    [-38.1747, -49.9296, -45.2733, -49.5571],
                                    [-19.7504, -23.9066,  -9.9139, -16.2088]]])
        >>> scores[:, 0, 1:] = MIN
        >>> scores.diagonal(0, 1, 2)[1:].fill_(MIN)
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> mst(scores, mask)
        tensor([[0, 2, 0, 2]])
    """

    _, seq_len, _ = scores.shape
    scores = scores.cpu().unbind()

    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length+1, :length+1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = MIN
            s = s.index_fill(1, torch.tensor(0), MIN)
            for root in roots:
                s[:, 0] = MIN
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)

    return pad(preds, total_length=seq_len).to(mask.device)

class MatrixTree(StructuredDistribution):
    r"""
    MatrixTree for calculating partitions and marginals of non-projective dependency trees in :math:`O(n^3)`
    by an adaptation of Kirchhoff's MatrixTree Theorem :cite:`koo-etal-2007-structured`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all possible dependent-head pairs.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking, regardless of root positions. Default: ``None``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Examples:
        >>> from supar import MatrixTree
        >>> batch_size, seq_len = 2, 5
        >>> lens = torch.tensor([3, 4])
        >>> arcs = torch.tensor([[0, 2, 0, 4, 2], [0, 3, 1, 0, 3]])
        >>> s1 = MatrixTree(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s2 = MatrixTree(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s1.max
        tensor([0.7174, 3.7910], grad_fn=<SumBackward1>)
        >>> s1.argmax
        tensor([[0, 0, 1, 1, 0],
                [0, 4, 1, 0, 3]])
        >>> s1.log_partition
        tensor([2.0229, 6.0558], grad_fn=<CopyBackwards>)
        >>> s1.log_prob(arcs)
        tensor([-3.2209, -2.5756], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([1.9711, 3.4497], grad_fn=<SubBackward0>)
        >>> s1.kl(s2)
        tensor([1.3354, 2.6914], grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        scores: torch.Tensor,
        lens: Optional[torch.LongTensor] = None,
        multiroot: bool = False
    ) -> MatrixTree:
        super().__init__(scores)

        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('__init__')
        
        batch_size, seq_len, *_ = scores.shape
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.index_fill(1, self.lens.new_tensor(0), 0)

        self.multiroot = multiroot

    def __repr__(self):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('__repr__')
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    def __add__(self, other):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('__add__')
        return MatrixTree(torch.stack((self.scores, other.scores)), self.lens, self.multiroot)

    @lazy_property
    def max(self):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('max')
        arcs = self.argmax
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    @lazy_property
    def argmax(self):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('argmax')
        with torch.no_grad():
            return mst(self.scores, self.mask, self.multiroot)

    def kmax(self, k: int) -> torch.Tensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('kmax')
        # TODO: Camerini algorithm
        raise NotImplementedError

    def sample(self):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('sample')
        raise NotImplementedError

    @lazy_property
    def entropy(self):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('entropy')
        return self.log_partition - (self.marginals * self.scores).sum((-1, -2))

    def cross_entropy(self, other: MatrixTree) -> torch.Tensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('cross_entropy')
        return other.log_partition - (self.marginals * other.scores).sum((-1, -2))

    def kl(self, other: MatrixTree) -> torch.Tensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('kl')
        return other.log_partition - self.log_partition + (self.marginals * (self.scores - other.scores)).sum((-1, -2))

    def score(self, value: torch.LongTensor, partial: bool = False) -> torch.Tensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('score')
        arcs = value
        if partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            scores = LogSemiring.zero_mask(self.scores, ~(arcs & mask))
            return self.__class__(scores, lens, **self.kwargs).log_partition
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    @torch.enable_grad()
    def forward(self, semiring: Semiring) -> torch.Tensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('forward')
        s_arc = self.scores
        batch_size, *_ = s_arc.shape
        mask, lens = self.mask.index_fill(1, self.lens.new_tensor(0), 1), self.lens
        # double precision to prevent overflows
        s_arc = semiring.zero_mask(s_arc, ~(mask.unsqueeze(-1) & mask.unsqueeze(-2))).double()

        # A(i, j) = exp(s(i, j))
        m = s_arc.view(batch_size, -1).max(-1)[0]
        A = torch.exp(s_arc - m.view(-1, 1, 1))

        # Weighted degree matrix
        # D(i, j) = sum_j(A(i, j)), if h == m
        #           0,              otherwise
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        # L(i, j) = D(i, j) - A(i, j)
        L = D - A
        if not self.multiroot:
            L.diagonal(0, 1, 2).add_(-A[..., 0])
            L[..., 1] = A[..., 0]
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(batch_size, 1, 1).masked_scatter_(mask.unsqueeze(-1), L[mask])
        L = L + nn.init.eye_(torch.empty_like(A[0])) * torch.finfo().tiny
        # Z = L^(0, 0), the minor of L w.r.t row 0 and column 0
        return (L[:, 1:, 1:].logdet() + m * lens).float()

class DependencyCRF(StructuredDistribution):
    r"""
    First-order TreeCRF for projective dependency trees :cite:`eisner-2000-bilexical,zhang-etal-2020-efficient`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all possible dependent-head pairs.
        lens (~torch.LongTensor): ``[batch_size]``.
            Sentence lengths for masking, regardless of root positions. Default: ``None``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Examples:
        >>> from supar import DependencyCRF
        >>> batch_size, seq_len = 2, 5
        >>> lens = torch.tensor([3, 4])
        >>> arcs = torch.tensor([[0, 2, 0, 4, 2], [0, 3, 1, 0, 3]])
        >>> s1 = DependencyCRF(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s2 = DependencyCRF(torch.randn(batch_size, seq_len, seq_len), lens)
        >>> s1.max
        tensor([3.6346, 1.7194], grad_fn=<IndexBackward>)
        >>> s1.argmax
        tensor([[0, 2, 3, 0, 0],
                [0, 0, 3, 1, 1]])
        >>> s1.log_partition
        tensor([4.1007, 3.3383], grad_fn=<IndexBackward>)
        >>> s1.log_prob(arcs)
        tensor([-1.3866, -5.5352], grad_fn=<SubBackward0>)
        >>> s1.entropy
        tensor([0.9979, 2.6056], grad_fn=<IndexBackward>)
        >>> s1.kl(s2)
        tensor([1.6631, 2.6558], grad_fn=<IndexBackward>)
    """

    def __init__(
        self,
        scores: torch.Tensor,
        lens: Optional[torch.LongTensor] = None,
        multiroot: bool = False
    ) -> DependencyCRF:
        super().__init__(scores)

        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('__init__')
        
        batch_size, seq_len, *_ = scores.shape
        self.lens = scores.new_full((batch_size,), seq_len-1).long() if lens is None else lens
        self.mask = (self.lens.unsqueeze(-1) + 1).gt(self.lens.new_tensor(range(seq_len)))
        self.mask = self.mask.index_fill(1, self.lens.new_tensor(0), 0)

        self.multiroot = multiroot

    def __repr__(self):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('__repr__')
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    def __add__(self, other):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('__add__')
        return DependencyCRF(torch.stack((self.scores, other.scores), -1), self.lens, self.multiroot)

    @lazy_property
    def argmax(self):
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('argmax')
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask, torch.where(self.backward(self.max.sum()))[2])

    def topk(self, k: int) -> torch.LongTensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('topk')
        preds = torch.stack([torch.where(self.backward(i))[2] for i in self.kmax(k).sum(0)], -1)
        return self.lens.new_zeros(*self.mask.shape, k).masked_scatter_(self.mask.unsqueeze(-1), preds)

    def score(self, value: torch.Tensor, partial: bool = False) -> torch.Tensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('score')
        arcs = value
        if partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            scores = LogSemiring.zero_mask(self.scores, ~(arcs & mask))
            return self.__class__(scores, lens, **self.kwargs).log_partition
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    def forward(self, semiring: Semiring) -> torch.Tensor:
        # print('/home/ke/Documents/projects/RNA/parser/my_rna_deep/supar/structs/tree.py')
        # print('forward')
        s_arc = self.scores
        batch_size, seq_len = s_arc.shape[:2]
        # [seq_len, seq_len, batch_size, ...], (h->m)
        s_arc = semiring.convert(s_arc.movedim((1, 2), (1, 0)))
        s_i = semiring.zeros_like(s_arc)
        s_c = semiring.zeros_like(s_arc)
        semiring.one_(s_c.diagonal().movedim(-1, 1))

        for w in range(1, seq_len):
            n = seq_len - w

            # [n, batch_size, ...]
            il = ir = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), 1)
            # INCOMPLETE-L: I(j->i) = <C(i->r), C(j->r+1)> * s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w).copy_(semiring.mul(il, s_arc.diagonal(-w).movedim(-1, 0)).movedim(0, -1))
            # INCOMPLETE-R: I(i->j) = <C(i->r), C(j->r+1)> * s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w).copy_(semiring.mul(ir, s_arc.diagonal(w).movedim(-1, 0)).movedim(0, -1))

            # [n, batch_size, ...]
            # COMPLETE-L: C(j->i) = <C(r->i), I(j->r)>, i <= r < j
            cl = semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), 1)
            s_c.diagonal(-w).copy_(cl.movedim(0, -1))
            # COMPLETE-R: C(i->j) = <I(i->r), C(r->j)>, i < r <= j
            cr = semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), 1)
            s_c.diagonal(w).copy_(cr.movedim(0, -1))
            if not self.multiroot:
                s_c[0, w][self.lens.ne(w)] = semiring.zero
        return semiring.unconvert(s_c)[0][self.lens, range(batch_size)]

