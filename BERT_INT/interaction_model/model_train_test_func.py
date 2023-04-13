import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from utils import *
import copy
from torch.nn import init

class MlP(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(MlP, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim, True)
        self.dense2 = nn.Linear(hidden_dim, 1, True)
        init.xavier_normal_(self.dense1.weight)
        init.xavier_normal_(self.dense2.weight)
    def forward(self,features):
        x = self.dense1(features)#[B,h]
        x = F.relu(x)
        x = self.dense2(x)#[B,1]
        x = F.tanh(x)
        x = torch.squeeze(x,1)#[B]
        return x

class Train_index_generator(object):
    def __init__(self, train_ill, train_candidate, entpair2f_idx, neg_num, batch_size):
        self.train_ill = train_ill
        self.train_candidate = copy.deepcopy(train_candidate)
        self.entpair2f_idx = entpair2f_idx
        self.iter_count = 0
        self.batch_size = batch_size
        self.neg_num = neg_num
        print("In Train_batch_index_generator, train_ILL num : {}".format(len(self.train_ill)))
        print("In Train_batch_index_generator, Batch size: {}".format(self.batch_size))
        print("In Train_batch_index_generator, Negative sampling num: {}".format(self.neg_num))
        for e in self.train_candidate.keys():
            self.train_candidate[e] = np.array(self.train_candidate[e])
        self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()

    def train_pair_index_gene(self):
        """
        generate training data (entity_index).
        """
        train_pair_indexs = []
        for pe1, pe2 in self.train_ill:
            neg_indexs = np.random.randint(len(self.train_candidate[pe1]),size=self.neg_num)
            ne2_list = self.train_candidate[pe1][neg_indexs].tolist()
            for ne2 in ne2_list:
                if ne2 == pe2:
                    continue
                ne1 = pe1
                train_pair_indexs.append((pe1, pe2, ne1, ne2))
                #(pe1,pe2) is aligned entity pair, (ne1,ne2) is negative sample
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        batch_num = int(np.ceil(len(train_pair_indexs) * 1.0 / self.batch_size))
        return train_pair_indexs, batch_num

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1
            batch_ids = self.train_pair_indexs[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
            pos_pairs = [(pe1, pe2) for pe1, pe2, ne1, ne2 in batch_ids]
            neg_pairs = [(ne1, ne2) for pe1, pe2, ne1, ne2 in batch_ids]

            pos_f_ids = [self.entpair2f_idx[pair_id] for pair_id in pos_pairs]
            neg_f_ids = [self.entpair2f_idx[pair_id] for pair_id in neg_pairs]
            return pos_f_ids, neg_f_ids
        else:
            self.iter_count = 0
            self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()
            raise StopIteration()


def one_step_train(Model, Optimizer, Criterion, Train_gene, f_emb, cuda_num):
    epoch_loss = 0
    for pos_f_ids, neg_f_ids in Train_gene:
        Optimizer.zero_grad()
        pos_feature = f_emb[torch.LongTensor(pos_f_ids)].cuda(cuda_num)
        neg_feature = f_emb[torch.LongTensor(neg_f_ids)].cuda(cuda_num)
        p_score = Model(pos_feature)
        n_score = Model(neg_feature)
        p_score = p_score.unsqueeze(-1)#[B,1]
        n_score = n_score.unsqueeze(-1)#[B,1]
        batch_size = p_score.shape[0]
        label_y = torch.ones(p_score.shape).cuda(cuda_num) #if y == 1 mean: p_score should ranked higher.
        batch_loss = Criterion(p_score, n_score, label_y)  #p_score > n_score
        epoch_loss += batch_loss.item() * batch_size
        batch_loss.backward()
        Optimizer.step()
    return epoch_loss

def compute_prec_rec_f1(aligns, truth_links):
    aligns = set(aligns)
    num_correct = len(aligns.intersection(truth_links))
    if num_correct == 0 or len(aligns) == 0:
        print("Got 0, 0, 0 in evaluation!!!")
        return 0, 0, 0
    precision = num_correct / len(aligns)
    recall = num_correct / len(truth_links)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def test(Model, test_candidate, test_ill, entpair2f_idx, f_emb, batch_size, cuda_num, test_topk):
    print(test_topk)
    test_ill_set = set(test_ill)
    test_pairs = []#all candidate entity pairs of Test set.
    for e1 in [a for a, b in test_ill]:
        for e2 in test_candidate[e1]:
            test_pairs.append((e1, e2))
    isin_test_ill_set_num = sum([pair in test_ill_set for pair in test_pairs])
    print("all test entity pair num {}/ max align entity pair num: {}".format(len(test_pairs), isin_test_ill_set_num))
    scores = []
    for start_pos in range(0, len(test_pairs), batch_size):
        batch_pair_ids = test_pairs[start_pos:start_pos + batch_size]
        batch_f_ids = [entpair2f_idx[pair_idx] for pair_idx in batch_pair_ids]
        batch_features = f_emb[torch.LongTensor(batch_f_ids)].cuda(cuda_num)  # [B,f]
        batch_scores = Model(batch_features)
        batch_scores = batch_scores.detach().cpu().tolist()
        scores.extend(batch_scores)
    assert len(test_pairs) == len(scores)
    # eval
    e1_to_e2andscores = dict()
    for i in range(len(test_pairs)):
        e1, e2 = test_pairs[i]
        score = scores[i]
        if (e1, e2) in test_ill_set:
            label = 1
        else:
            label = 0
        if e1 not in e1_to_e2andscores:
            e1_to_e2andscores[e1] = []
        e1_to_e2andscores[e1].append((e2, score, label))

    all_test_num = len(e1_to_e2andscores.keys()) # test set size.
    result_labels = []
    for e, value_list in e1_to_e2andscores.items():
        v_list = value_list
        v_list.sort(key=lambda x: x[1], reverse=True)
        label_list = [label for e2, score, label in v_list]
        label_list = label_list[:test_topk]
        result_labels.append(label_list)
        del label_list
    result_labels = np.array(result_labels)

    print(result_labels.shape)

    count_found = 0
    for i in range(len(result_labels)):
        index = np.where(result_labels[i] == 1)[0]
        if len(index) > 0:
            count_found += 1

    result_labels = result_labels.sum(axis=0).tolist()
    topk_list = []
    for i in range(test_topk):
        nums = sum(result_labels[:i + 1])
        topk_list.append(round(nums / all_test_num, 5))
    print("hit @ 1: {:.5f}    hit @ 5: {:.5f}    hit @10 : {:.5f}    ".format(topk_list[1 - 1]*100,topk_list[5 - 1]*100,topk_list[10 - 1]*100), end="")
    if test_topk >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_list[25 - 1]*100), end="")
    if test_topk >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_list[50 - 1]*100), end="")
    print("")

    # MRR
    MRR = 0
    MR = 0
    counter_found = 0
    for i in range(len(result_labels)):
        MR += (i + 1) * result_labels[i]
        MRR += (1 / (i + 1)) * result_labels[i]
    print("MR_without_norm:", MR)
    print("MRR_without_norm:", MRR)
    print(all_test_num)
    print("found:", count_found)
    MRR /= all_test_num
    MR /= all_test_num
    print("MR:", MR)
    print("MRR:", MRR)
    # Save the best entity e2 in KG2 for each e1 in KG1 (entity matchers alignment)
    best_align = {}
    for i in range(len(test_pairs)):
        e1, e2 = test_pairs[i]
        score = scores[i]
        if e1 in best_align:
            if score > best_align[e1][1]:
                best_align[e1] = (e2, score)
        else:
            best_align[e1] = (e2, score)
    best_pairs = [(e1, e2) for e1, (e2, _) in best_align.items()]
    # assert len(best_pairs) == len(test_ill) + len(ent_out_1)
    return best_pairs

def read_truth(path):
    truth = []
    with open(path) as f:
        for l in f:
            (e1, e2) = l.rstrip("\n").split("\t")
            truth.append((int(e1), int(e2)))
    return truth

def train(Model, Optimizer, Criterion, Train_gene, f_emb_list, test_candidate, test_ill,
          entpair2f_idx, data_path, epoch_num, eval_num, cuda_num, test_topk):
    feature_emb = torch.FloatTensor(f_emb_list)
    print("start training interaction model!")
    for epoch in range(epoch_num):
        start_time = time.time()
        epoch_loss = one_step_train(Model, Optimizer, Criterion, Train_gene, feature_emb, cuda_num)
        print("Epoch {} loss {:.4f} using time {:.3f}".format(epoch, epoch_loss, time.time() - start_time))
        if (epoch + 1) % eval_num == 0 and epoch != 0 :
            start_time = time.time()
            best_pairs = test(Model, test_candidate, test_ill, entpair2f_idx, feature_emb, 2048, cuda_num, test_topk)
            truth = read_truth(data_path + 'ref_pairs')

            # precision, recall, f1 = compute_prec_rec_f1(best_pairs, truth)
            # print("Final test result:")
            # print("\tPrecision:", precision)
            # print("\tRecall:", recall)
            # print("\tF1:", f1)
            print("test using time {:.3f}".format(time.time() - start_time))