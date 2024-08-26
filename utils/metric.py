
from __future__ import annotations
from typing import Dict, List, Optional, Tuple


import torch

class Metric(object):

    def __init__(self,args, reverse: Optional[bool] = None, eps: float = 1e-12 )-> Metric:

        super().__init__()

        self.n = 0.0
        self.count = 0.0
        self.total_arc_loss = 0.0
        self.total_rel_loss = 0.0
        self.total_loss = 0.0
        self.reverse = reverse
        self.eps = eps
        self.args = args

    def __repr__(self):
        
        arc_loss, rel_loss, total_loss = self.loss

        return f"loss: {total_loss:.4f} - " + ' '.join([f"{key}: {val:6.2%}" for key, val in self.values.items()])

    def __lt__(self, other: Metric) -> bool:

        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score < other.score) if not self.reverse else (self.score > other.score)

    def __le__(self, other: Metric) -> bool:

        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score <= other.score) if not self.reverse else (self.score >= other.score)

    def __gt__(self, other: Metric) -> bool:

        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score > other.score) if not self.reverse else (self.score < other.score)

    def __ge__(self, other: Metric) -> bool:

        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score >= other.score) if not self.reverse else (self.score <= other.score)

    def __add__(self, other: Metric) -> Metric:
        return other

    @property
    def score(self):
        raise AttributeError

    @property
    def loss(self):
        return self.total_arc_loss/(self.count + self.eps), self.total_rel_loss/(self.count + self.eps), self.total_loss/(self.count + self.eps) 


    @property
    def values(self):
        raise AttributeError


class AttachmentMetric(Metric):

    def __init__(

        self,
        args,
        arc_loss: Optional[float] = None,
        rel_loss: Optional[float] = None,
        loss: Optional[float] = None,

        preds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        golds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.BoolTensor] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> AttachmentMetric:
        self.args = args
        super().__init__(args, reverse=reverse, eps=eps)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

        self.correct_stack_sum = torch.tensor([]).to(device)
        self.total_preds_as_stack_sum = torch.tensor([]).to(device)
        self.total_actual_stack_sum = torch.tensor([]).to(device)

        if loss is not None:
            self(arc_loss, rel_loss,loss, preds, golds, mask)

    def __call__(

        self,
        arc_loss: float,
        rel_loss: float,
        loss: float,
        preds: Tuple[torch.Tensor, torch.Tensor],
        golds: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.BoolTensor
    ) -> AttachmentMetric:
        

        self.num_rows = mask.shape[0]

        arc_preds, rel_preds, arc_golds, rel_golds = *preds, *golds

        rel_map = self.args.relation_dic

        relation = 'stem'
        rel_id = rel_map[relation]

        correct_stack = ((rel_preds == rel_id) & (rel_golds == rel_id) & (arc_preds == arc_golds) & mask).type(torch.int)
        total_preds_as_stack = ((rel_preds == rel_id) & mask).type(torch.int)
        total_actual_stack = ((rel_golds == rel_id) & mask).type(torch.int)

        correct_stack_sum = correct_stack.sum(1) 
        total_preds_as_stack_sum = total_preds_as_stack.sum(1)
        total_actual_stack_sum = total_actual_stack.sum(1)

        self.correct_stack_sum = torch.cat((self.correct_stack_sum, correct_stack_sum), 0)
        self.total_preds_as_stack_sum = torch.cat((self.total_preds_as_stack_sum, total_preds_as_stack_sum), 0)
        self.total_actual_stack_sum = torch.cat((self.total_actual_stack_sum, total_actual_stack_sum), 0)


        self.n += len(mask)
        self.count += 1
        self.total_arc_loss += float(arc_loss)
        self.total_rel_loss += float(rel_loss)
        self.total_loss += float(loss)


        return self

    def __add__(self, other: AttachmentMetric) -> AttachmentMetric:

        metric = AttachmentMetric(args=self.args, eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count

        metric.total_arc_loss = self.total_arc_loss + other.total_arc_loss
        metric.total_rel_loss = self.total_rel_loss + other.total_rel_loss
        metric.total_loss = self.total_loss + other.total_loss

        metric.correct_stack_sum = torch.cat((self.correct_stack_sum, other.correct_stack_sum), 0)
        metric.total_preds_as_stack_sum = torch.cat((self.total_preds_as_stack_sum, other.total_preds_as_stack_sum), 0)
        metric.total_actual_stack_sum = torch.cat((self.total_actual_stack_sum, other.total_actual_stack_sum), 0)

        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):

        stack_precision = (self.correct_stack_sum / self.total_preds_as_stack_sum.clamp(min=1)).mean().item()
        stack_recall = (self.correct_stack_sum / self.total_actual_stack_sum.clamp(min=1)).mean().item()
        stack_f1 = 2 * stack_precision * stack_recall / (stack_precision + stack_recall + self.eps)

        return stack_f1

    @property
    def stack_pre(self):

        stack_precision = self.correct_stack_sum / self.total_preds_as_stack_sum.clamp(min=1)

        return stack_precision.mean().item()

    @property
    def stack_recall(self):

        stack_recall = self.correct_stack_sum / self.total_actual_stack_sum.clamp(min=1)

        return stack_recall.mean().item()

    @property
    def stack_f1(self):
        precision = self.stack_pre
        recall = self.stack_recall
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @property
    def values(self) -> Dict:
        return {'stack_pre': self.stack_pre,
                'stack_recall': self.stack_recall,
                'stack_f1': self.stack_f1
        }

def evaluate_result(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a) * torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    return precision, recall, f1_score