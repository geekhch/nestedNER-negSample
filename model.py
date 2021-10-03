import numpy as np

import torch
from torch import nn
from pytorch_pretrained_bert import BertModel

from misc import flat_list
from misc import iterative_support, conflict_judge
from utils import UnitAlphabet, LabelAlphabet, GPU_ID


class PhraseClassifier(nn.Module):

    def __init__(self,
                 lexical_vocab: UnitAlphabet,
                 label_vocab: LabelAlphabet,
                 hidden_dim: int,
                 dropout_rate: float,
                 neg_rate: float,
                 bert_path: str,
                 bert_cache_dir):
        super(PhraseClassifier, self).__init__()

        self._lexical_vocab = lexical_vocab
        self._label_vocab = label_vocab
        self._neg_rate = neg_rate
        self.bunble_size = 1000

        self._encoder = BERT(bert_path, cache_dir=bert_cache_dir)
        _dist_size = 16
        self._distance_embedding = nn.Embedding(512, _dist_size)
        self._classifier = MLP(self._encoder.dimension * 4 + _dist_size, hidden_dim, len(label_vocab), dropout_rate)
        self._criterion = nn.NLLLoss()

    def forward(self, var_h, **kwargs):
        pass

    def itertive_span(self, positions, bert_repr):
        """
        :param positions: Longtensors (sample_id, left_idx, right_idx)
        :return:
        """
        if torch.cuda.is_available():
            positions = positions.cuda(GPU_ID)
        left_repr = bert_repr[positions[:,0], positions[:,1]]
        right_repr = bert_repr[positions[:,0], positions[:,2]]
        distance_idx = positions[:,2] - positions[:, 1]
        distance_repr = self._distance_embedding(distance_idx)
        _repr = torch.cat([left_repr, right_repr, left_repr-right_repr,left_repr*right_repr, distance_repr], dim=-1)
        return self._classifier(_repr)


    def _pre_process_input(self, utterances):
        lengths = [len(s) for s in utterances]
        max_len = max(lengths)
        pieces = iterative_support(self._lexical_vocab.tokenize, utterances)
        units, positions = [], []

        for tokens in pieces:
            units.append(flat_list(tokens))
            cum_list = np.cumsum([len(p) for p in tokens]).tolist()
            positions.append([0] + cum_list[:-1])

        sizes = [len(u) for u in units]
        max_size = max(sizes)
        cls_sign = self._lexical_vocab.CLS_SIGN
        sep_sign = self._lexical_vocab.SEP_SIGN
        pad_sign = self._lexical_vocab.PAD_SIGN
        pad_unit = [[cls_sign] + s + [sep_sign] + [pad_sign] * (max_size - len(s)) for s in units]
        starts = [[ln + 1 for ln in u] + [max_size + 1] * (max_len - len(u)) for u in positions]

        var_unit = torch.LongTensor([self._lexical_vocab.index(u) for u in pad_unit])
        attn_mask = torch.LongTensor([[1] * (lg + 2) + [0] * (max_size - lg) for lg in sizes])
        var_start = torch.LongTensor(starts)

        if torch.cuda.is_available():
            var_unit = var_unit.cuda(GPU_ID)
            attn_mask = attn_mask.cuda(GPU_ID)
            var_start = var_start.cuda(GPU_ID)
        return var_unit, attn_mask, var_start, lengths  # var_start是每个sub-word的偏移位置

    def _pre_process_output(self, entities, lengths):
        '''生成训练的正负span样本'''
        positions, labels = [], []
        batch_size = len(entities)

        for utt_i in range(0, batch_size):
            for segment in entities[utt_i]:
                positions.append((utt_i, segment[0], segment[1]))
                labels.append(segment[2])

        for utt_i in range(0, batch_size):
            reject_set = [(e[0], e[1]) for e in entities[utt_i]]
            s_len = lengths[utt_i]
            neg_num = int(s_len * self._neg_rate) + 1

            candies = flat_list([[(i, j) for j in range(i, s_len) if (i, j) not in reject_set] for i in range(s_len)])
            if len(candies) > 0:
                sample_num = min([neg_num, len(candies), 4 * (len(reject_set) + 1)])
                assert sample_num > 0

                np.random.shuffle(candies)
                for i, j in candies[:sample_num]:
                    positions.append((utt_i, i, j))
                    labels.append("O")

        var_lbl = torch.LongTensor(iterative_support(self._label_vocab.index, labels))
        if torch.cuda.is_available():
            var_lbl = var_lbl.cuda(GPU_ID)
        return torch.LongTensor(positions), var_lbl

    def estimate(self, sentences, segments):
        var_sent, attn_mask, start_mat, lengths = self._pre_process_input(sentences)
        bert_repr = self._encoder(var_sent, attn_mask, start_mat)
        score_t = []
        # score_t = self(var_sent, mask_mat=attn_mask, starts=start_mat)

        positions, targets = self._pre_process_output(segments, lengths)
        for bunble in range(0, len(positions), self.bunble_size):
            score_t.append(self.itertive_span(positions[bunble: bunble+self.bunble_size], bert_repr))

        flat_s = torch.cat(score_t, dim=0)
        return self._criterion(torch.log_softmax(flat_s, dim=-1), targets)

    def inference(self, sentences, allow_conflit=False):
        '''
        allow_conflit: 允许相交的嵌套实体存在, 但预测为同类别的实体不可相交
        '''
        var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
        bert_repr = self._encoder(var_sent, attn_mask, starts)
        batch_size = len(lengths)
        positions = [(utt_i, i, j) for utt_i in range(batch_size) for i in range(lengths[utt_i]) for j in range(i, lengths[utt_i])]
        positions = torch.LongTensor(positions)
        if torch.cuda.is_available():
            positions = positions.cuda(GPU_ID)

        candidates = [[] for _ in range(batch_size)]

        for bunble in range(0, len(positions), self.bunble_size):
            bunble_data = positions[bunble:bunble+self.bunble_size]
            log_items = self.itertive_span(bunble_data, bert_repr)
            score_items = torch.log_softmax(log_items, dim=-1)
            val_items, idx_items = torch.max(score_items, dim=-1)
            listing_it = idx_items.cpu().numpy().tolist()  # 预测最大类别对应的idx
            listing_vt = val_items.cpu().numpy().tolist()  # 预测最大类别对应的概率值
            label_items = iterative_support(self._label_vocab.get, listing_it)
            for pos, lb, vt in zip(bunble_data, label_items, listing_vt):
                if lb != 'O':
                    candidates[pos[0]].append((pos[1], pos[2], lb, vt))


        entities = []
        for segments in candidates:
            ordered_seg = sorted(segments, key=lambda e: -e[-1])  # 分值从高到低排序
            filter_list = []
            for elem in ordered_seg:
                flag = False
                current = (elem[0], elem[1])  # [start, end, label]
                for prior in filter_list:
                    if allow_conflit and prior[2] != elem[2]:
                        continue
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append((elem[0].item(), elem[1].item(), elem[2]))
            entities.append(sorted(filter_list, key=lambda e: e[0]))
        return entities


class BERT(nn.Module):

    def __init__(self, source_path, cache_dir):
        super(BERT, self).__init__()
        self._repr_model = BertModel.from_pretrained(source_path, cache_dir=cache_dir)

    @property
    def dimension(self):
        return 768

    @property
    def position_embedding_weight(self):
        return self._repr_model.embeddings.position_embeddings.weight.detach()

    def forward(self, var_h, attn_mask, starts):
        all_hidden, _ = self._repr_model(var_h, attention_mask=attn_mask, output_all_encoded_layers=False)

        batch_size, _, hidden_dim = all_hidden.size()
        _, unit_num = starts.size()
        positions = starts.unsqueeze(-1).expand(batch_size, unit_num, hidden_dim)
        return torch.gather(all_hidden, dim=-2, index=positions)  # 沿着某一个维度上使用给定的索引取出对应的值。


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()

        self._activator = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, output_dim))
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, var_h):
        return self._activator(self._dropout(var_h))
