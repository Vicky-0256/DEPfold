import copy
import json
import os
import time
import fm
import tempfile
from typing import Iterable, Union
from datetime import datetime

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
from utils.utils import LinearLR, seed_everything, ProgressBar, init_logger, logger, write_prediction_results
from utils.metric import AttachmentMetric, Metric, evaluate_result
from modules.tokenizer import TransformerTokenizer
from modules.pretrained import TransformerEmbedding
from DataProcessing.dataset_dep import Biaffine_Dataset, read_examples
from models.biaffine_supar import BiaffineModel
import numpy as np

from config import Config
import RNA
import argparse

class RNAbiaffine():

    def __init__(self, args):
        self.args = args
        self.model = BiaffineModel(args)

    def train(self, clip: float = 5.0, amp: bool = False, update_steps: int = 1):
        logger.info("Loading the data")

        train_dataset = Biaffine_Dataset(
            self.args,
            examples=read_examples(self.args, file_path="/home/ke/Documents/RNA/mxfold2-data/data/bpRNA_dataset-canonicals/TR0/", session=self.args.train_session),
            
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/RNAstrAlign/train1/", session=self.args.train_session),
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/TAB/TrainSetA/", session=self.args.train_session),
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/test/", session=self.args.train_session),
            data_type="train"
        )
        eval_dataset = Biaffine_Dataset(
            self.args,
            examples=read_examples(self.args, file_path="/home/ke/Documents/RNA/mxfold2-data/data/bpRNA_dataset-canonicals/TS0/", session=self.args.eval_session),
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/RNAstrAlign/val1/", session=self.args.eval_session),
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/TAB/TestSetB/", session=self.args.eval_session),
            data_type="dev"
        )
        test_dataset = Biaffine_Dataset(
            self.args,
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA/mxfold2-data/data/bpRNA_dataset-canonicals/VL0/", session=self.args.eval_session),
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/ArchiveII/", session=self.args.test_session),
            examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/TAB/TestSetB/", session=self.args.eval_session),
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/check/", session=self.args.test_session),
            # examples=read_examples(self.args, file_path="/home/ke/Documents/RNA_parser/RNA_parser/data/RNAstrAlign/test1/", session=self.args.test_session),
            data_type="test"
        )

        train_iter = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_gpu_train_batch_size, collate_fn=train_dataset._create_collate_fn(), num_workers=8)
        eval_iter = DataLoader(eval_dataset, shuffle=False, batch_size=self.args.per_gpu_eval_batch_size, collate_fn=eval_dataset._create_collate_fn(), num_workers=4)
        test_iter = DataLoader(test_dataset, shuffle=False, batch_size=self.args.per_gpu_eval_batch_size, collate_fn=test_dataset._create_collate_fn(), num_workers=4)

        logger.info(f"train: {train_iter}")
        logger.info(f"dev: {eval_iter}")
        logger.info(f"test: {test_iter}")

        self.args.steps = len(train_iter) * self.args.num_train_epochs // update_steps

        self.model.to(self.args.device)
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.scaler = GradScaler(enabled=amp)

        # self.best_metric = Metric(self.args)
        self.best_f1 = 0

        for epoch in range(int(self.args.num_train_epochs)):
            start = datetime.now()

            logger.info(f"Epoch {epoch + 1} / {self.args.num_train_epochs}:")
            logger.info("***** Running train *****")

            self.model.train()

            batch_loss = []
            batch_arc_loss = []
            batch_rel_loss = []

            pbar = ProgressBar(n_total=len(train_iter), desc='Training')

            for step, batch in enumerate(train_iter):
                batch = (batch[0],) + tuple(t.to(self.args.device) for t in batch[1:])
                arc_loss, rel_loss, loss = self.train_step(batch)

                loss.backward()

                self.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(True)

                batch_loss.append(loss.item())
                batch_arc_loss.append(arc_loss.item())
                batch_rel_loss.append(rel_loss.item())

                pbar(step, {
                    'loss': sum(batch_loss) / len(batch_loss),
                    'arc_loss': sum(batch_arc_loss) / len(batch_arc_loss),
                    'rel_loss': sum(batch_rel_loss) / len(batch_rel_loss)
                })

            # evaluate
            logger.info("***** Running Evaluation *****")
            self.model.eval()

            # metric = Metric(self.args)
            # test_metric = Metric(self.args)
            beta = self.args.beta

            eval_results, run_time =[],[]
            test_results, test_time =[],[]
            for batch in eval_iter:
                batch = (batch[0],) + tuple(t.to(self.args.device) for t in batch[1:])
                # metric += self.eval_step(batch, beta)
                interval_t, eval_result = self.eval_step(batch, beta)
                run_time.append(interval_t)
                eval_results += eval_result
            p, r, eval_f1 = zip(*eval_results)

            print('Eval F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}\n'.format(np.average(eval_f1), np.average(p), np.average(r), np.average(run_time)))
            logging.info('Eval F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}'.format(
                np.average(eval_f1), np.average(p), np.average(r), np.average(run_time))
            )
            for batch in test_iter:
                batch = (batch[0],) + tuple(t.to(self.args.device) for t in batch[1:])
                # test_metric += self.eval_step(batch, beta)
                interval_t, eval_result = self.eval_step(batch, beta)
                test_time.append(interval_t)
                test_results += eval_result

            p, r, f1 = zip(*test_results)
            print('Test F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}\n'.format(np.average(f1), np.average(p), np.average(r), np.average(test_time)))
            logging.info('Test F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}\n'.format(np.average(f1), np.average(p), np.average(r), np.average(test_time)))
            # logger.info(f"dev: {metric}")
            # logger.info(f"test: {test_metric}")
            t = datetime.now() - start
            # if metric >= self.best_metric:
            if np.average(eval_f1) >= self.best_f1:
                early_stop = 0
                best_model = copy.deepcopy(self.model.module if hasattr(self.model, "module") else self.model)
                torch.save(best_model.state_dict(), os.path.join(self.args.output_dir, "model.pt"))
                print('eval_f1 >= self.best_f1')
                # print('metric', metric.values)
                # self.best_e, self.best_metric = epoch, metric
                self.best_e, self.best_f1 = epoch, np.average(eval_f1)
            else:
                early_stop += 1
                # logger.info(f"{t}s elapsed\n")
                if early_stop == self.args.early_stop:
                    # logger.info(f"Early stop in {epoch} epoch!")
                    print(f"Early stop in {epoch} epoch!")
                    break

    # def predict(self):
    #     args = self.args
    #     path = self.args.path
    #     self.model.load_state_dict(torch.load(path, map_location='cpu'))
    #     self.model.to(self.args.device)

    #     logger.info("Loading the data")
    #     dataset = Biaffine_Dataset(self.args, examples=read_examples(self.args, file_path=self.args.predict, session=self.args.predict_session), data_type="test")
    #     pred_iter = DataLoader(dataset, shuffle=False, batch_size=self.args.per_gpu_eval_batch_size, collate_fn=dataset._create_collate_fn(), num_workers=4)

    #     relation_dic = self.args.relation_dic

    #     logger.info("Making predictions on the data")
    #     start = datetime.now()
    #     self.model.eval()
    #     seq_all, gold, predict = [], [], []

    #     for batch in pred_iter:
    #         batch = (batch[0],) + tuple(t.to(self.args.device) for t in batch[1:])

    #         seq, batch_token_ids, mask, arcs, rels = batch
    #         mask[:, 0] = 0
    #         s_arc, s_rel = self.model(batch_token_ids=batch_token_ids)

    #         s_time = time.time()
    #         beta = args.beta
    #         seed = -1
    #         pred_contacts = self.model.decode(s_arc, s_rel, seed, beta, mask, self.args.tree, self.args.proj).cpu()
    #         interval_t = time.time() - s_time

    #         contacts = self.create_contacts_from_arcs_and_rels(arcs, rels)
            
    #         pred_result = list(map(lambda i: evaluate_result(pred_contacts()[i],
    #                                                                     contacts()[i]), range(contacts.shape[0])))
    #         gold_structure = self.contacts_to_dot_bracket(contacts)
    #         pred_structure = self.contacts_to_dot_bracket(pred_contacts)

    #         seq_all.extend(seq)
    #         gold.extend(gold_structure)
    #         predict.extend(pred_structure)



    #     p, r, f1 = zip(*pred_result)
    #     print('predict file F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}\n'.format(np.average(f1), np.average(p), np.average(r), np.average(interval_t)))
        
    #     predict_path = self.args.predict_save
    #     with open(predict_path + '/seq.txt', 'w') as file:
    #         for s in seq:
    #             file.write(s + '\n')
    #     with open(predict_path + '/predict.txt', 'w') as file:
    #         for structure in predict:
    #             file.write(structure + '\n')
    #     with open(predict_path + '/gold.txt', 'w') as file:
    #         for structure in gold:
    #             file.write(structure + '\n')

    def predict(self):
        args = self.args
        path = self.args.path
        # self.model.load_state_dict(torch.load(path, map_location='cpu'))
        state_dict = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.args.device)

        logger.info("Loading the data")
        dataset = Biaffine_Dataset(args, examples=read_examples(args, file_path=args.predict, session=args.predict_session), data_type="test")
        pred_iter = DataLoader(dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size, collate_fn=dataset._create_collate_fn(), num_workers=4)

        logger.info("Making predictions on the data")
        start = datetime.now()
        self.model.eval()
        seq_all, gold, predict = [], [], []
        pred_result = []

        for batch in pred_iter:
            batch = (batch[0],) + tuple(t.to(args.device) for t in batch[1:])

            seq, batch_token_ids, mask, arcs, rels = batch
            mask[:, 0] = 0
            
            with torch.no_grad():
                s_arc, s_rel = self.model(batch_token_ids=batch_token_ids)

                s_time = time.time()
                print(s_time)
                beta = args.beta
                seed = -1
                pred_contacts = self.model.decode(s_arc, s_rel, seed, beta, mask, args.tree, args.proj)
                interval_t = time.time() - s_time
                # torch.set_printoptions(edgeitems=pred_contacts.size(-1), 
                #        linewidth=1000)
                # print(pred_contacts)

                contacts = self.create_contacts_from_arcs_and_rels(arcs, rels)
                # print(arcs)
                # print(rels)
                # print(contacts)


                # Move tensors to CPU for evaluation
                pred_contacts_cpu = pred_contacts.cpu()
                contacts_cpu = contacts.cpu()

                for i in range(contacts.shape[0]):
                    try:
                        result = evaluate_result(pred_contacts_cpu[i], contacts_cpu[i])
                        pred_result.append(result)
                    except Exception as e:
                        logger.error(f"Error in evaluate_result for item {i}: {e}")
                        continue

                gold_structure = self.contacts_to_dot_bracket(contacts_cpu)
                pred_structure = self.contacts_to_dot_bracket(pred_contacts_cpu)

                seq_all.extend(seq)
                gold.extend(gold_structure)
                predict.extend(pred_structure)

        p, r, f1 = zip(*pred_result)
        logger.info('Predict file F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}\n'.format(
            np.mean(f1), np.mean(p), np.mean(r), np.mean(interval_t)))
        
        predict_path = args.predict_save
        with open(os.path.join(predict_path, 'seq.txt'), 'w') as file:
            file.write('\n'.join(seq_all))
        with open(os.path.join(predict_path, 'predict.txt'), 'w') as file:
            file.write('\n'.join(predict))
        with open(os.path.join(predict_path, 'gold.txt'), 'w') as file:
            file.write('\n'.join(gold))

        logger.info(f"Prediction completed. Time taken: {datetime.now() - start}")


    # def contacts_to_dot_bracket(self, contacts):
    #     """
    #     将接触矩阵转换为点括号表示法的 RNA 二级结构。
        
    #     参数:
    #     contacts (torch.Tensor): 形状为 [batch_size, seq_len, seq_len] 的接触矩阵

    #     返回:
    #     list of str: 每个元素是一个序列的点括号表示
    #     """
    #     batch_size, seq_len, _ = contacts.shape
    #     dot_bracket_list = []

    #     for batch in range(batch_size):
    #         structure = ['.' for _ in range(seq_len)]
    #         stack = []

    #         for i in range(seq_len):
    #             if torch.any(contacts[batch, i, i+1:]):
    #                 j = torch.where(contacts[batch, i, i+1:])[0][0].item() + i + 1
    #                 structure[i] = '('
    #                 structure[j] = ')'
    #                 stack.append(j)
    #             elif stack and i == stack[-1]:
    #                 stack.pop()
    #             else:
    #                 structure[i] = '.'

    #         dot_bracket_list.append(''.join(structure))

    #     return dot_bracket_list

    def contacts_to_dot_bracket(self, contacts):
        """
        将接触矩阵转换为点括号表示法的 RNA 二级结构，改进版本。
        
        参数:
        contacts (torch.Tensor): 形状为 [batch_size, seq_len, seq_len] 的接触矩阵

        返回:
        list of str: 每个元素是一个序列的点括号表示
        """
        batch_size, seq_len, _ = contacts.shape
        dot_bracket_list = []

        for batch in range(batch_size):
            structure = ['.' for _ in range(seq_len)]
            stack = []

            for i in range(seq_len):
                if i in stack:
                    continue
                
                if torch.any(contacts[batch, i, i+1:]):
                    j = torch.where(contacts[batch, i, i+1:])[0][0].item() + i + 1
                    
                    # 检查 j 是否已经被配对
                    if j in stack:
                        print('j 已经被配对')

                    if j not in stack:
                        structure[i] = '('
                        structure[j] = ')'
                        stack.append(j)


                
            dot_bracket_list.append(''.join(structure))

        return dot_bracket_list


        '''
        for step, batch in enumerate(pred_iter):

            batch = (batch[0],) + tuple(t.to(self.args.device) for t in batch[1:])
            batch_seq, batch_token_ids, mask, batch_arcs, batch_rels = batch
            lens = [(t.sum(dim=0) - 2).tolist() for t in mask]
            mask = torch.logical_and(batch_token_ids.ne(1), batch_token_ids.ne(2))
            mask[:, 0] = 0

            s_arc, s_rel = self.model(batch_token_ids=batch_token_ids)
            batch_arcs = [i.tolist() for i in batch_arcs[mask].split(lens)]
            batch_rels = [i.tolist() for i in batch_rels[mask].split(lens)]

            seq.extend(batch_seq)
            arcs.extend(batch_arcs)
            rels.extend(batch_rels)

            gold_all_structures, gold_energies = self.calculate_stem_energies(batch_arcs, batch_rels, relation_dic, batch_seq)
            gold.extend(gold_all_structures)

            beta = args.beta
            best_arc, best_rel = None, None

            for seed in range(args.decode_round):
                if args.decode_round == 1:
                    seed = -1

                batch_arc_preds, batch_rel_preds = self.model.decode(s_arc, s_rel, seed, beta, mask, tree=True, proj=True)
                batch_arc_preds = [i.tolist() for i in batch_arc_preds[mask].split(lens)]
                batch_rel_preds = [i.tolist() for i in batch_rel_preds[mask].split(lens)]

                all_structures, energies = self.calculate_stem_energies(batch_arc_preds, batch_rel_preds, relation_dic, batch_seq)

                if seed == 0 or seed == -1:
                    best_arc = batch_arc_preds
                    best_rel = batch_rel_preds
                    best_energies = energies
                    best_structures = all_structures
                else:
                    for i in range(len(energies)):
                        if energies[i] < best_energies[i]:
                            best_energies[i] = energies[i]
                            best_structures[i] = all_structures[i]
                            best_arc[i] = batch_arc_preds[i]
                            best_rel[i] = batch_rel_preds[i]

            predict.extend(best_structures)
            arc_preds.extend(best_arc)
            rel_preds.extend(best_rel)

        predict_path = self.args.predict_save
        self.write_results(seq, predict, gold, arcs, rels, arc_preds, rel_preds, relation_dic, predict_path)

    def write_results(self, seq, predict, gold, arcs, rels, arc_preds, rel_preds, relation_dic, predict_path):
        with open(predict_path + '/seq.txt', 'w') as file:
            for s in seq:
                file.write(s + '\n')
        with open(predict_path + '/predict.txt', 'w') as file:
            for structure in predict:
                file.write(structure + '\n')
        with open(predict_path + '/gold.txt', 'w') as file:
            for structure in gold:
                file.write(structure + '\n')

        self.process_and_write_relations(arcs, rels, relation_dic, predict_path, type='gold')
        self.process_and_write_relations(arc_preds, rel_preds, relation_dic, predict_path, type='predict')

    def process_and_write_relations(self, arcs, rels, relation_dic, predict_path, type):
        file_names = {name: f"{predict_path}/{name}_{type}.txt" for name in relation_dic}
        files_content = {name: [] for name in file_names}

        for arc_line, rel_line in zip(arcs, rels):
            line_lengths = len(arc_line)
            line_representation = {name: [] for name in file_names}
            temp = ['.' for _ in range(line_lengths)]

            for i, (arc, rel) in enumerate(zip(arc_line, rel_line)):
                for rel_name, rel_val in relation_dic.items():
                    if rel == rel_val:
                        target_idx = arc - 1
                        if rel_name == 'stem':
                            if i < target_idx:
                                temp[i] = '('
                                temp[target_idx] = ')'
                            else:
                                temp[i] = ')'
                                temp[target_idx] = '('
                        line_representation[rel_name].append(f"[{i + 1},{target_idx + 1}]")
            for name in file_names:
                files_content[name].append(''.join(line_representation[name]))

        for name, content in files_content.items():
            with open(file_names[name], 'w') as file:
                for line in content:
                    file.write(line + '\n')

    def calculate_stem_energies(self, arcs, rels, relation_dic, sequences):
        all_structures = []
        energies = []
        for arc_line, rel_line, sequence in zip(arcs, rels, sequences):
            line_lengths = len(arc_line)
            temp = ['.' for _ in range(line_lengths)]
            for i, (arc, rel) in enumerate(zip(arc_line, rel_line)):
                for rel_name, rel_val in relation_dic.items():
                    if rel == rel_val and rel_name == 'stem':
                        target_idx = arc - 1
                        if target_idx < 0 or target_idx >= len(temp) or i >= len(temp):
                            continue
                        if i < target_idx:
                            temp[i] = '('
                            temp[target_idx] = ')'
                        else:
                            temp[i] = ')'
                            temp[target_idx] = '('

            structure = ''.join(temp)
            all_structures.append(structure)
            if not self.is_balanced_and_nested_correctly(structure):
                energy = 1e7
            else:
                fc = RNA.fold_compound(sequence)
                try:
                    energy = fc.eval_structure(structure)
                except Exception as e:
                    print(f"Error processing structure: {structure}, setting energy to infinity.")
                    energy = 1e7
            energies.append(energy)
        return all_structures, energies

    def is_balanced_and_nested_correctly(self, structure):
        balance = 0
        for char in structure:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            if balance < 0:
                return False
        return balance == 0
    '''

    def train_step(self, batch):
        seq, batch_token_ids, mask, arcs, rels = batch
        mask[:, 0] = 0
        s_arc, s_rel = self.model(batch_token_ids=batch_token_ids)
        arc_loss, rel_loss, loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
        return arc_loss, rel_loss, loss
    
    def create_contacts_from_arcs_and_rels(self, arcs, rels):

        batch_size, seq_len = arcs.shape
        device = arcs.device 

        stem_index = self.args.relation_dic['stem']
        pseudo_index = self.args.relation_dic['pseudo']
        
        # 创建一个全零的contacts矩阵
        contacts = torch.zeros((batch_size, seq_len, seq_len), device=device)
        
        # 使用高效的张量操作来填充contacts矩阵
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len,device=device).unsqueeze(0).expand(batch_size, -1)
        

        stem_mask = (rels == stem_index)
        pseudo_mask = (rels == pseudo_index)
        combined_mask = stem_mask | pseudo_mask
        # combined_mask = pseudo_mask

        
        # 设置contacts矩阵中对应的位置为1
        valid_indices = combined_mask & (batch_indices < batch_size)
        
        contacts[batch_indices[valid_indices], seq_indices[valid_indices], arcs[valid_indices]] = 1
        # contacts = contacts + contacts.transpose(1, 2)
        
        return contacts

    def eval_step(self, batch, beta):
        seq, batch_token_ids, mask, arcs, rels = batch
        mask[:, 0] = 0
        s_arc, s_rel = self.model(batch_token_ids=batch_token_ids)
        arc_loss, rel_loss, loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
        seed = -1
        # arc_preds, rel_preds = self.model.decode(s_arc, s_rel, seed, beta, mask, self.args.tree, self.args.proj)
        # return AttachmentMetric(self.args, arc_loss, rel_loss, loss, (arc_preds, rel_preds), (arcs, rels), mask)
        s_time = time.time()

        pred_contacts = self.model.decode(s_arc, s_rel, seed, beta, mask, self.args.tree, self.args.proj)
        interval_t = time.time() - s_time

        contacts = self.create_contacts_from_arcs_and_rels(arcs, rels)
        
        # eval_result = list(map(lambda i: evaluate_result(pred_contacts()[i],
        #                                                              contacts()[i]), range(contacts.shape[0])))

        eval_result = list(map(lambda i: evaluate_result(pred_contacts[i],
                                                        contacts[i]), range(contacts.shape[0])))




        return interval_t, eval_result
        

    def clip_grad_norm_(self, params: Union[Iterable[torch.Tensor], torch.Tensor], max_norm: float, norm_type: float = 2) -> torch.Tensor:
        self.scaler.unscale_(self.optimizer)
        return nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    def init_optimizer(self) -> Optimizer:
        return AdamW(
            params=[{'params': p, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)} for n, p in self.model.named_parameters()],
            lr=self.args.lr,
            betas=(getattr(self.args, 'mu', 0.9), getattr(self.args, 'nu', 0.999)),
            eps=getattr(self.args, 'eps', 1e-8),
            weight_decay=getattr(self.args, 'weight_decay', 0)
        )

    def init_scheduler(self) -> _LRScheduler:
        return LinearLR(
            optimizer=self.optimizer,
            warmup_steps=getattr(self.args, 'warmup_steps', int(self.args.steps * getattr(self.args, 'warmup', 0))),
            steps=self.args.steps
        )

def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--embedding", type=str, default='roberta-base', help="one-hot, RNA-fm,roberta-base")
    parser.add_argument("--finetune", action='store_true', help="finetune embedding")

    parser.add_argument("--train_session", default="TR0", type=str, help="TR0, TR1")
    parser.add_argument("--eval_session", default="TS0", type=str, help="VL0")
    parser.add_argument("--test_session", default="VB", type=str, help="TS0, bpnew")

    parser.add_argument("--cache_data", default="./data/bp_", type=str, help="data pkl path")
    parser.add_argument("--is_pse", action='store_true', help="include pseudoknot or not")    
    parser.add_argument("--per_gpu_train_batch_size", default=3, type=int, help="训练Batch size的大小")
    parser.add_argument("--per_gpu_eval_batch_size", default=3, type=int, help="验证Batch size的大小")
    parser.add_argument("--num_train_epochs", default=100, type=float, help="训练轮数")
    parser.add_argument("--early_stop", default=8, type=int, help="早停")

    parser.add_argument("--lr", default=5.0e-05, type=float, help="训练轮数")
    parser.add_argument("--lr_rate", default=20, type=float, help="学习率比例")

    parser.add_argument("--tree", action='store_true', help="tree")
    parser.add_argument("--proj", action='store_true', help="proj")

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")

    parser.add_argument("--output_dir", default="./output", type=str, help="保存模型的路径")

    parser.add_argument("--mode", default='train', type=str, help="train or predict")
    parser.add_argument("--loss", default='cross_entropy', type=str, help="cross_entropy or focal_loss")

    # when predict
    
    parser.add_argument("--predict", default="/home/ke/Documents/RNA_parser/RNA_parser/data/bp_/", type=str, help="predict data file path")
    # parser.add_argument("--predict", default="/home/ke/Documents/RNA_parser/RNA_parser/data/Str_", type=str, help="predict data file path")
    parser.add_argument("--predict_session", default="VLA", type=str, help="predict session")
    parser.add_argument("--decode_round", default=1, type=int, help="how many times to make the prediction")
    parser.add_argument("--beta", default=0.0, type=float, help="beta stem map to added to arc")
    parser.add_argument("--predict_save", default="/home/ke/Documents/RNA_parser/RNA_parser/test/test/", type=str, help="result save")
    parser.add_argument("--path", default="/home/ke/Documents/RNA_parser/RNA_parser/TABoutput/Roberta_experiment/TRA/model.pt", type=str, help="model path")

    return parser

def main():
    parser = get_argparse()
    args = parser.parse_args()
    args.time = time.strftime("%m-%d_%H:%M", time.localtime())
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))

    seed_everything(args.seed)

    if not os.path.exists(args.output_dir):
        print(f'mkdir {args.output_dir}')
        os.mkdir(args.output_dir)
    if not os.path.exists(args.cache_data):
        print(f'mkdir {args.cache_data}')
        os.makedirs(args.cache_data)
    if not hasattr(args, 'relation_dic'):
        setattr(args, 'relation_dic', {'loop': 1, 'root': 2, 'stem': 3, 'stemnect': 4, 'pseudo': 5})
    print(args.relation_dic)
    log_file_path = os.path.join(args.output_dir, "{}.log".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    init_logger(log_file_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.embedding == 'RNA-fm':
        args.encoder, args.tokenizer = fm.pretrained.rna_fm_t12(model_location="pretrained/RNA-FM_pretrained.pth")
    elif args.embedding == 'roberta-base':
        args.tokenizer = TransformerTokenizer(name=args.embedding)
        args.encoder = TransformerEmbedding
    else:
        raise ValueError(f"Unsupported embedding type: {args.embedding}")

    args.embedding_concat = False

    if args.mode == 'train':
        RNAbiaffine(args=args).train()
    elif args.mode == 'predict':
        RNAbiaffine(args=args).predict()

if __name__ == "__main__":
    main()
