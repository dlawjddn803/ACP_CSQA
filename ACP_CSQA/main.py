import logging
import warnings
import argparse

from ACP_CSQA.model_params import *
from train import GraphC_QAModel

warnings.filterwarnings(action='ignore')

logger = logging.getLogger('gct_dual')

PARAMS_MAP = {
    'ACB_dual':AMR_CN_BERT_PARAMS,
}

def train_model(args, local_rank):
    args = PARAMS_MAP[args.encoder_type]
    if 'google' in args['lm_model']:
        lm_args = args['lm_model'][args['lm_model'].index('/')+1:]
    else:
        lm_args = args['lm_model']
    args['prefix'] = str(args['encoder_type']) + '_' + args['task'] + '_' + args['feature'] + '_lr' + str(
        args['lr']) + '_' + str(args['batch_multiplier']) + '_' + lm_args

    assert len(args['cnn_filters']) % 2 == 0

    model = GraphC_QAModel(args, local_rank)
    model.train()

def evaluate_model(args, local_rank):
    eval_file = args.eval_file
    gpus = args.gpus
    args = PARAMS_MAP[args.encoder_type]
    if 'google' in args['lm_model']:
        lm_args = args['lm_model'][args['lm_model'].index('/')+1:]
    else:
        lm_args = args['lm_model']
    args['prefix'] = str(args['encoder_type']) + '_' + args['task'] + '_' + args['feature'] + '_lr' + str(
        args['lr']) + '_' + str(args['batch_multiplier']) + '_' + lm_args

    assert len(args['cnn_filters']) % 2 == 0

    model = GraphC_QAModel(args, local_rank)
    model.evaluate_model(eval_file, gpus)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='gct_dual')
    parser.add_argument('--mode', dest="mode", type=str, default='eval')
    parser.add_argument("--encoder_type", dest="encoder_type", type=str, default=None,
                            help="Model Name")
    parser.add_argument("--eval_file", dest='eval_file', type=str, default=None)
    parser.add_argument("--gpus", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = get_params()

    if args.mode == 'train':
        train_model(args, 0)
    else:
        evaluate_model(args, 0)

