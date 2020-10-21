import time
import math
import copy
import csv
import collections
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch
import logging
import random

import numpy as np
from tqdm import tqdm, trange

from optimization import WarmupLinearSchedule, AdamW
from data import Vocab, DataLoader, STR, END, CLS, SEL, TL, rCLS
from prepare.extract_property_train import LexicalMap
from utils import move_to_cuda, EarlyStopping

from bert_models.modeling_bert import BertForMultipleChoice
from models.reasoningCT import Reasoning_AMR_CN_DUAL

# Reasoning_AMR, Reasoning_CN

from transformers import (
    BertConfig,
    BertTokenizer)

from tensorboardX import SummaryWriter

import os

warnings.filterwarnings(action='ignore')


# os.environ["CUDA_VISIBLE_DEVICES"]="2" #실행 파일에 추가!

logger = logging.getLogger('gct_dual')
# logger.setLevel(logging) # 여기서 로깅 레벨 수정할 것


stream_hander = logging.StreamHandler()
logger.addHandler(stream_hander)

MODEL_CLASSES = {
    'ACB_dual':(BertConfig, BertForMultipleChoice, BertTokenizer),
    }


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if int(n_gpu[0]) > 0:
        torch.cuda.manual_seed_all(seed)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class GraphC_QAModel(object):
    def __init__(self, args, local_rank):
        super(GraphC_QAModel, self).__init__()
        self.save_args = args
        args['cnn_filters'] = list(zip(args['cnn_filters'][:-1:2], args['cnn_filters'][1::2]))
        args = collections.namedtuple("HParams", sorted(args.keys()))(**args)
        if not os.path.exists(args.ckpt):
            os.mkdir(args.ckpt)
        self.args = args
        self.local_rank = local_rank


    def _build_model(self):
        print(self.args, 'final')
        self.device = torch.device("cuda:"+str(self.args.gpus[0]) if torch.cuda.is_available() else "cpu")
        print(self.device, 'here')
        vocabs, lexical_mappings = [], []
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.encoder_type]
        self.bert_config = config_class.from_pretrained(
            self.args.lm_model,
        )
        self.bert_tokenizer = tokenizer_class.from_pretrained(
            self.args.lm_model
        )

        if self.args.bert_pretrained_file == None:

            self.bert_model = model_class.from_pretrained(
                self.args.lm_model,
                config=self.args.lm_model
            ).to(self.device)

        else:
            self.bert_model = model_class.from_pretrained(
                self.args.bert_pretrained_file,
            ).to(self.device)
            print('bert_pretrained')
        # self.device = torch.device('cuda', self.args.gpus[0])
        if self.args.encoder_type in ['ACB_dual']:
            vocabs, lexical_mapping = self._prepare_data()
            self.model = Reasoning_AMR_CN_DUAL(vocabs,
                                               self.args.concept_char_dim, self.args.concept_dim,
                                               self.args.cnn_filters, self.args.char2concept_dim,
                                               self.args.rel_dim, self.args.rnn_hidden_size, self.args.rnn_num_layers,
                                               self.args.embed_dim, self.args.bert_embed_dim, self.args.ff_embed_dim,
                                               self.args.num_heads,
                                               self.args.dropout,
                                               self.args.snt_layer,
                                               self.args.graph_layers,
                                               self.args.pretrained_file, self.device, self.args.batch_size,
                                               self.args.lm_model, self.bert_config, self.bert_model, self.bert_tokenizer, self.args.bert_max_length,
                                               self.args.n_answers,
                                               self.args.encoder_type,
                                               self.args.max_conceptnet_length,
                                               self.args.conceptnet_path,
            )

        else:
            pass


        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        return vocabs, lexical_mapping

    def _update_lr(self, optimizer, embed_size, steps, warmup_steps):
        for param_group in optimizer.param_groups:
            param_group['lr'] = embed_size ** -0.5 * min(steps ** -0.5, steps * (warmup_steps ** -1.5))

    def _average_gradients(self, model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size

    def _prepare_data(self):

        vocabs = dict()
        vocabs['concept'] = Vocab(self.args.concept_vocab, 5, [CLS])
        vocabs['token'] = Vocab(self.args.token_vocab, 5, [STR, END])
        vocabs['token_char'] = Vocab(self.args.token_char_vocab, 100, [STR, END])
        vocabs['concept_char'] = Vocab(self.args.concept_char_vocab, 100, [STR, END])
        vocabs['relation'] = Vocab(self.args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
        lexical_mapping = LexicalMap()

        for name in vocabs:
            print((name, vocabs[name].size, vocabs[name].coverage))
        return vocabs, lexical_mapping

    def train(self):

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        task = self.args.task
        tb_writer = SummaryWriter(log_dir='./runs/'+task+"/"+current_time+self.args.prefix, comment=self.args.prefix)

        vocabs, lexical_mapping = self._build_model()

        train_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.train_data, self.args.batch_size,
                                for_train=True)
        dev_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.dev_data, self.args.batch_size,
                              for_train=False)
        test_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.test_data, self.args.batch_size,
                               for_train='Eval')

        train_data.set_unk_rate(self.args.unk_rate)

        # WRITE PARAMETERS
        with open('./' + 'param' + '.txt', 'w') as f:

            for name, param in self.model.named_parameters():
                f.writelines('name:'+name+"\n")
                f.writelines(str(param))
                f.writelines('size:'+str(param.size())+'\n')

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        gradient_accumulation_steps = 1
        t_total = len(train_data) // gradient_accumulation_steps * self.args.epochs

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)

        self.model.zero_grad()

        set_seed(42, self.args.gpus)

        batches_acm, loss_acm = 0, 0

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Task: %s", self.args.task)
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", self.args.epochs)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Running Language Model = %s", self.args.lm_model)
        logger.info("  Running Model = %s", self.args.encoder_type)


        best_acc = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        total_steps = 0

        train_iterator = trange(int(self.args.epochs), desc="Epoch")

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_data, desc="Iteration")

            running_loss = 0.0
            running_corrects = 0

            batch_count = self.args.batch_multiplier

            # Turn on the train mode
            for step, batch in enumerate(epoch_iterator):

                self.model.train()
                batch = move_to_cuda(batch, self.device)

                logits, labels, ans_ids = self.model(batch, train=True)
                logits_for_pred = logits.clone().detach()
                loss = self.criterion(logits, labels)
                loss_value = loss.item()

                pred_values, pred_indices = torch.max(logits_for_pred, 1)
                labels = labels.tolist()
                pred = pred_indices.tolist()
                corrects = [i for i, j in zip(labels, pred) if i == j]

                # Statistics
                running_loss += loss.item()
                running_corrects += len(corrects)

                if batch_count == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                    total_steps += 1
                    optimizer.zero_grad()
                    self.model.zero_grad()

                    batch_count = self.args.batch_multiplier


                loss_acm += loss_value

                loss.backward()
                batch_count -= 1


                if (batches_acm % (self.args.batch_multiplier*self.args.batch_size) == 0) & (batches_acm != 0) & (step != 0):
                    logger.info('Train Epoch %d, Batch %d, loss %.3f, Accuracy %.3f',
                                _, batches_acm, loss_acm / batches_acm, running_corrects / (self.args.batch_size*step))
                    tb_writer.add_scalar('Training_loss', loss_acm / batches_acm, batches_acm)
                    tb_writer.add_scalar('Training_Accuracy', running_corrects / (self.args.batch_size*step))
                    torch.cuda.empty_cache()
                batches_acm += 1

            epoch_loss = running_loss / batches_acm
            epoch_acc = running_corrects / len(train_data)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                _, epoch_loss, epoch_acc)
            )

            tb_writer.add_scalar('Training_Epoch_loss', epoch_loss, _)
            tb_writer.add_scalar('Training_Epoch_Accuracy', epoch_acc, _)

            # Evaluate on Development Set
            eval_epoch_acc, eval_epoch_loss = self._run_evaluate(dev_data, _, write_answer=False)

            print('Overall_Dev Acc: {:.4f}'.format(
                eval_epoch_acc)
            )

            tb_writer.add_scalar('Dev_Epoch_Accuracy', eval_epoch_acc, _)

            ##################################

            # Evaluate on Test Set
            test_epoch_acc, test_epoch_loss = self._run_evaluate(test_data, _, write_answer=True)

            print('Overall_Test Acc: {:.4f}'.format(
                test_epoch_acc)
            )
            tb_writer.add_scalar('Test_Epoch_Accuracy', test_epoch_acc, _)

            # Save only best accuracy model on dev set
            if eval_epoch_acc > best_acc:
                best_acc = eval_epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(epoch_acc, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            self.model.train()

        logger.info('Best val Acc: {:4f}'.format(best_acc))

        torch.save({'args': self.save_args, 'model': best_model_wts},
                   '%s/epoch%d_batch%d_model_best_%s' % (self.args.ckpt, self.args.epochs, batches_acm, self.args.prefix))

    def _run_evaluate(self, dev_data, epoch, write_answer=False):
        running_corrects = 0
        eval_loss_sum, batch_acm = 0, 0
        answer_tempelate = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

        self.model.eval()
        if write_answer:
            with open(self.args.prefix+'epoch_'+str(epoch)+ '.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                       quoting=csv.QUOTE_MINIMAL)

                with torch.no_grad():
                    for step, batch in enumerate(dev_data):

                        batch = move_to_cuda(batch, self.device)
                        eval_logits, eval_labels, ans_ids, = self.model(batch, train=False)
                        eval_logits_forpred = eval_logits.clone().detach()

                        eval_loss = self.criterion(eval_logits, eval_labels)
                        loss_value = eval_loss.item()

                        pred_values, pred_indices = torch.max(eval_logits_forpred, 1)

                        eval_labels = eval_labels.tolist()
                        eval_pred = pred_indices.tolist()

                        corrects = [i for i, j in zip(eval_labels, eval_pred) if i == j]
                        eval_loss_sum += eval_loss
                        batch_acm += 1

                        # Statistics
                        running_corrects += len(corrects)

                        for i, pred in enumerate(eval_pred):
                            csvwriter.writerow([ans_ids[i], answer_tempelate[int(pred_indices[i])]])
                    print('Overall accuracy: ', (running_corrects / len(dev_data)))

                    epoch_acc = running_corrects / len(dev_data)
                    epoch_loss = eval_loss_sum / batch_acm
        else:
            with torch.no_grad():
                for step, batch in enumerate(dev_data):
                    batch = move_to_cuda(batch, self.device)
                    eval_logits, eval_labels, ans_ids, = self.model(batch, train=False)
                    eval_logits_forpred = eval_logits.clone().detach()

                    eval_loss = self.criterion(eval_logits, eval_labels)
                    loss_value = eval_loss.item()

                    pred_values, pred_indices = torch.max(eval_logits_forpred, 1)

                    eval_labels = eval_labels.tolist()
                    eval_pred = pred_indices.tolist()

                    corrects = [i for i, j in zip(eval_labels, eval_pred) if i == j]
                    eval_loss_sum += eval_loss
                    batch_acm += 1

                    # Statistics
                    running_corrects += len(corrects)

                epoch_acc = running_corrects / len(dev_data)
                epoch_loss = eval_loss_sum / batch_acm

        return epoch_acc, epoch_loss

    def evaluate_model(self, eval_file, gpus):
        self.device = torch.device("cuda:" + str(gpus) if torch.cuda.is_available() else "cpu")
        print('device', self.device)
        test_models = []
        if os.path.isdir(eval_file):
            for file in os.listdir(eval_file):
                fname = os.path.join(eval_file, file)
                if os.path.isfile(fname):
                    test_models.append(fname)
            model_args = torch.load(fname, map_location=self.device)['args']
        else:
            test_models.append(eval_file)
            model_args = torch.load(eval_file, map_location=self.device)['args']

        from data import Vocab, DataLoader, STR, END, CLS, SEL, TL, rCLS
        model_args = collections.namedtuple("HParams", sorted(model_args.keys()))(**model_args)
        vocabs = dict()
        vocabs['concept'] = Vocab(model_args.concept_vocab, 5, [CLS])
        vocabs['token'] = Vocab(model_args.token_vocab, 5, [STR, END])
        vocabs['token_char'] = Vocab(model_args.token_char_vocab, 100, [STR, END])
        vocabs['concept_char'] = Vocab(model_args.concept_char_vocab, 100, [STR, END])
        vocabs['relation'] = Vocab(model_args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
        lexical_mapping = LexicalMap()

        if self.args.encoder_type in ['ACB_dual', 'ACX_dual', 'ACA_dual', 'ACE_dual', 'ACBL_dual', 'ACEL_dual']:
            vocabs, lexical_mapping = self._prepare_data()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.encoder_type]

            bert_config = config_class.from_pretrained(
                self.args.lm_model,
            )
            bert_tokenizer = tokenizer_class.from_pretrained(
                self.args.lm_model
            )
            bert_model = model_class.from_pretrained(
                self.args.lm_model,
                from_tf=bool(".ckpt" in self.args.lm_model),
                config=self.args.lm_model,
            ).to(self.device)

            eval_model = Reasoning_AMR_CN_DUAL(vocabs,
                                               model_args.concept_char_dim, model_args.concept_dim,
                                               model_args.cnn_filters, model_args.char2concept_dim,
                                               model_args.rel_dim, model_args.rnn_hidden_size, model_args.rnn_num_layers,
                                               model_args.embed_dim, model_args.bert_embed_dim, model_args.ff_embed_dim,
                                               model_args.num_heads,
                                               model_args.dropout,
                                               model_args.snt_layer,
                                               model_args.graph_layers,
                                               model_args.pretrained_file, self.device, model_args.batch_size,
                                               model_args.lm_model, bert_config, bert_model, bert_tokenizer, model_args.bert_max_length,
                                               model_args.n_answers,
                                               model_args.encoder_type,
                                               model_args.gcn_concept_dim, model_args.gcn_hidden_dim, model_args.gcn_output_dim, model_args.max_conceptnet_length,
                                               model_args.conceptnet_path,

            )

        else:
            eval_model = ''
        test_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.test_data, model_args.batch_size,
                               for_train='Eval')

        answer_tempelate = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        # Evaluate!
        logger.info("***** Running Evaluating *****")
        logger.info("  Task: %s", self.args.task)
        logger.info("  Num examples = %d", len(test_data))
        logger.info("  Running Language Model = %s", model_args.lm_model)
        logger.info("  Running Model = %s", model_args.encoder_type)
        logger.info("  Running File = %s", eval_file)
        logger.info("  Test data = %s", self.args.test_data)

        for test_model in test_models:
            eval_model.load_state_dict(torch.load(test_model, map_location=self.device)['model'])
            eval_model = eval_model.cuda(self.device)
            eval_model.eval()

            running_corrects = 0
            eval_loss_sum, batch_acm = 0, 0
            with open(test_model + model_args.prefix + '.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                       quoting=csv.QUOTE_MINIMAL)
                for batch in test_data:
                    batch = move_to_cuda(batch, self.device)
                    eval_logits, eval_labels, ans_ids, = eval_model(batch, train=False)
                    eval_logits_forpred = eval_logits.clone().detach()

                    pred_values, pred_indices = torch.max(eval_logits_forpred, 1)
                    eval_labels = eval_labels.tolist()
                    eval_pred = pred_indices.tolist()

                    corrects = [i for i, j in zip(eval_labels, eval_pred) if i == j]

                    batch_acm += 1
                    # Statistics
                    running_corrects += len(corrects)
                    for i, pred in enumerate(eval_pred):
                        csvwriter.writerow([ans_ids[i], answer_tempelate[int(pred_indices[i])]])
                print('Overall accuracy: ', (running_corrects / len(test_data)))


