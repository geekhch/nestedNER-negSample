import argparse
import os
import json
import re

import torch
from pytorch_pretrained_bert import BertAdam

from utils import UnitAlphabet, LabelAlphabet
from model import PhraseClassifier
from misc import fix_random_seed
from utils import corpus_to_iterator, Procedure, split_long_passage, GPU_ID

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


def predict_a_long_text(text, model, batch_size, src_info=None):
    '''text: 一段长长的文字...'''
    sentence_segs = split_long_passage(text)
    material = [{'sentence': str(list(sent)), 'labeled entities':str([])} for sent in sentence_segs]
    test_loader = corpus_to_iterator(None, batch_size, False, material=material)
    ans = Procedure.test_for_cluener(model, test_loader, allow_conflit=True)
    return ans

def predict_for_clue_submit(args, model):
    fix_random_seed(args.random_state)
    test_loader = corpus_to_iterator(os.path.join(args.data_dir, "test.json"), args.batch_size, False)
    ans = Procedure.test_for_cluener(model, test_loader, True)
    with open(os.path.join(args.check_dir, 'cluener_predict.json'), 'w', encoding='utf8') as f:
        for jobj in ans:
            f.write(json.dumps(jobj, ensure_ascii=False) + '\n')

@app.route('/', methods=['POST', 'GET'])
def predict_service():
    try:
        text = request.values.get('text')
        if not text:
            raise Exception("使用post方法且text字段不能为空！")
        
        ########## 模型预测
        results = predict_a_long_text(text, model, batch_size=args.batch_size)
        
        ret = {
            'result': results,
            'code' : 200
        }
    except Exception as e:
        from traceback import print_exc
        print_exc()
        ret = {
            'code': 500,
            'message': str(e)
        }
    return jsonify(ret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-dd", type=str, required=True)
    parser.add_argument("--check_dir", "-cd", type=str, required=True)
    parser.add_argument("--resource_dir", "-rd", type=str, required=True)
    parser.add_argument("--random_state", "-rs", type=int, default=0)
    parser.add_argument("--epoch_num", "-en", type=int, default=40)
    parser.add_argument("--batch_size", "-bs", type=int, default=16)

    parser.add_argument("--negative_rate", "-nr", type=float, default=0.7)
    parser.add_argument("--warmup_proportion", "-wp", type=float, default=0.1)
    parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)
    parser.add_argument("--cache_dir", "-cache", type=str, default='resource/bert')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=True), end="\n\n")

    checkpoint_path = os.path.join(args.check_dir, "model.pt")
    model = torch.load(checkpoint_path, map_location='cpu')
    model = model.cuda(GPU_ID) if torch.cuda.is_available() else model.cpu()
    app.run(debug=False, host='0.0.0.0', port=16060)
    # predict_for_clue_submit(args, model)
