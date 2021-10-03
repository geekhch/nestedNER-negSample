from typing import Dict
from tqdm import tqdm
import time
import re

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer

from misc import extract_json_data
from misc import iob_tagging, f1_score
from datetime import datetime

GPU_ID = torch.cuda.device_count() - 1

class UnitAlphabet(object):

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"
    PAD_SIGN, UNK_SIGN = "[PAD]", "[UNK]"

    def __init__(self, source_path, cache_dir):
        self._tokenizer = BertTokenizer.from_pretrained(source_path, do_lower_case=True, cache_dir=cache_dir)

    def tokenize(self, item):
        return self._tokenizer.tokenize(item)

    def index(self, items):
        return self._tokenizer.convert_tokens_to_ids(items)

def strftime():
    return datetime.now().strftime('%m-%d_%H-%M')

class LabelAlphabet(object):

    def __init__(self):
        super(LabelAlphabet, self).__init__()

        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


def corpus_to_iterator(file_path, batch_size, if_shuffle, label_vocab=None, material=None):
    if material is None:
        material = extract_json_data(file_path)
    instances = [(eval(e["sentence"]), eval(e["labeled entities"])) for e in material]

    if label_vocab is not None:
        label_vocab.add("O")
        for _, u in instances:
            for _, _, l in u:
                label_vocab.add(l)

    class _DataSet(Dataset):

        def __init__(self, elements):
            self._elements = elements

        def __getitem__(self, item):
            return self._elements[item]

        def __len__(self):
            return len(self._elements)

    def distribute(elements):
        sentences, entities = [], []
        for s, e in elements:
            sentences.append(s)
            entities.append(e)
        return sentences, entities

    wrap_data = _DataSet(instances)
    return DataLoader(wrap_data, batch_size, if_shuffle, collate_fn=distribute)


def split_long_passage(passage, max_seq_len=256):
    """
    input a long text, cut it into senteces.
    output: batch_data[Dict]: {model_inputs:{}, map_locs}
    """

    sentence_list, tmp_sentence = [], []

    # 分句，去掉空白符
    for i in range(len(passage)):
        token = passage[i]
        if not re.match('\s', token) and len(tmp_sentence) < max_seq_len - 2:
            tmp_sentence.append(token)

        if token in ['。', '!', '\n'] or (token == ',' and len(tmp_sentence) > max_seq_len - 30):
            if len(tmp_sentence) > 1:
                sentence_list.append(tmp_sentence.copy())
            tmp_sentence.clear()

    if len(tmp_sentence) > 1:
        sentence_list.append(tmp_sentence.copy())
    return sentence_list


class Procedure(object):
    @staticmethod
    def train(model, dataset, optimizer):
        model.train()
        time_start, total_penalties, total_sample = time.time(), 0.0, 0

        for batch in tqdm(dataset, ncols=50):
            loss = model.estimate(*batch)

            total_penalties += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_con = time.time() - time_start
        return total_penalties, time_con

    @staticmethod
    def test(model, dataset):
        model.eval()
        time_start = time.time()
        tp, fp, fn = 0.0, 0.0, 0.0
        for sentences, segments in tqdm(dataset, ncols=50):
            with torch.no_grad():
                predictions = model.inference(sentences)
            gold_set, pred_set = set(), set()
            for segs_1, segs_2 in zip(predictions, segments):
                segs_1, segs_2 = set(segs_1), set(segs_2)
                tp += len(segs_1 & segs_2)
                fn += len(segs_2 - segs_1)
                fp += len(segs_1 - segs_2)
        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)
        out_f1 = 2 * p * r / (p + r + 1e-7)
        return out_f1, time.time() - time_start

    @staticmethod
    def test_for_cluener(model, dataset, allow_conflit) -> list:
        model.eval()
        time_start = time.time()
        ans = []
        idx = 0
        for sentences, segments in tqdm(dataset, ncols=50):
            batch_size = len(sentences)
            with torch.no_grad():
                predictions = model.inference(sentences, allow_conflit)

            for i in range(batch_size):
                text = ''.join(sentences[i])
                sample = {
                    'id': idx,
                    'text': text,
                    'label': {}
                }
                idx += 1
                for e_start, e_end, e_type in predictions[i]:
                    e_text = text[e_start: e_end+1]
                    if e_type not in sample['label']:
                        sample['label'][e_type] = {}
                    if e_text not in sample['label'][e_type]:
                        sample['label'][e_type][e_text] = []
                    sample['label'][e_type][e_text].append([e_start, e_end])
                ans.append(sample)
        return ans
