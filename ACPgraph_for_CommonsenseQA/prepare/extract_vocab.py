#!/usr/bin/env python
#coding: utf-8
from smatch import AMR
from AMRGraph import AMRGraph
from collections import Counter
import json
from AMRGraph import  _is_abs_form
from multiprocessing import Pool


class LexicalMap(object):

    def __init__(self):
        pass

    def get(self, concept, vocab=None):
        # concept들의 리스트
        cp_seq = []
        for conc in concept:
            cp_seq.append(conc)

        if vocab is None:
            return cp_seq

        # predictable_token이 들어오는데, (data_loader할 때) 그때 없는 애들 추가해준다.
        new_tokens = set(cp for cp in cp_seq if vocab.token2idx(cp) == vocab.unk_idx)
        token2idx, idx2token = dict(), dict()
        nxt = vocab.size
        for x in new_tokens:
            token2idx[x] = nxt
            idx2token[nxt] = x
            nxt +=1

        return cp_seq, token2idx, idx2token

class AMRIO:
    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()

                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                elif line.startswith('# ::tokens '):
                    tokens = json.loads(line[len('# ::tokens '):])
                    tokens = [ to if _is_abs_form(to) else to.lower() for to in tokens]
                elif line.startswith('# ::lemmas '):
                    lemmas = json.loads(line[len('# ::lemmas '):])
                    lemmas = [le if _is_abs_form(le) else le.lower() for le in lemmas]
                elif line.startswith('# ::pos_tags '):
                    pos_tags = json.loads(line[len('# ::pos_tags '):])
                elif line.startswith('# ::ner_tags '):
                    ner_tags = json.loads(line[len('# ::ner_tags '):])
                elif line.startswith('# ::abstract_map '):
                    abstract_map = json.loads(line[len('# ::abstract_map '):])
                elif line.startswith('# ::option '):
                    option = line[len('# ::option '):]
                    option = ast.literal_eval(option)
                elif line.startswith('# ::answer_key '):
                    answer_key = line[len('# ::answer_key '):]
                # elif line.startswith('# ::answer '):
                #     answer = line[len('# ::answer '):]
                elif line.startswith('# ::save-date '):
                    graph_line = AMR.get_amr_line(f)
                    raw_amr = AMR.parse_AMR_line(graph_line)
                    myamr = AMRGraph(raw_amr)

                    yield tokens, lemmas, abstract_map, myamr, answer_key, option, amr_id, sentence, raw_amr

def read_file(filename):
    # read preprocessed amr file
    token, lemma, abstract, amrs, answer_keys, options, amr_ids, sentences, raw_amrs = [], [], [], [], [], [], [], [], []


    for _tok, _lem, _abstract, _myamr, _answer_key, _option, _amr_id, _sentence, _raw_amr in AMRIO.read(filename):
        token.append(_tok)
        lemma.append(_lem)
        abstract.append(_abstract)
        amrs.append(_myamr)
        answer_keys.append(_answer_key)
        options.append(_option)
        amr_ids.append(_amr_id)
        sentences.append(_sentence)
        raw_amrs.append(_raw_amr)

    print ('read from %s, %d amrs'%(filename, len(token)))
    return amrs, token, lemma, abstract, answer_keys, options, amr_ids, sentences, raw_amrs

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))



import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--amr_files', type=str, nargs='+')
    parser.add_argument('--nprocessors', type=int, default=4)
    parser.add_argument('--extend', type=bool, default=False)
    parser.add_argument('--option_amr', type=bool, default=False)
    parser.add_argument('--concept_seed', type=str)
    parser.add_argument('--answer_len', type=int, default=5)
    parser.add_argument('--omcs', type=bool, default=False)
    return parser.parse_args()

import ast

if __name__ == "__main__":
    # collect concepts and relations

    args = parse_config()

    # print(cn)

    amrs, token, lemma, abstract, answer_keys, options, amr_ids, sentences, raw_amrs = read_file(args.train_data)
    lexical_map = LexicalMap()


    def work(data):
        amr, lem, tok = data
        concept, depth, relation, ok, ARGs = amr.collect_concepts_and_relations()
        assert ok, "not connected"
        lexical_concepts = set(lexical_map.get(concept))

        return concept, depth, relation, ARGs

    pool = Pool(args.nprocessors)

    if args.extend == True:
        res = pool.map(work, zip(amrs, lemma, token), len(amrs) // args.nprocessors)
    else:
        res = pool.map(work, zip(amrs, lemma, token), len(amrs) // args.nprocessors)

    tot_pairs = 0
    multi_path_pairs = 0
    tot_paths = 0
    extreme_long_paths = 0
    avg_path_length = 0.
    conc, rel = [], []

    for concept, depth, relation, ARGs in res:
        conc.append(concept)

        for x in relation:
            for y in relation[x]:
                tot_pairs += 1
                if len(relation[x][y]) > 1:
                    multi_path_pairs += 1
                for path in relation[x][y]:
                    tot_paths += 1
                    path_len = path['length']
                    rel.append(path['edge'])
                    if path_len > 8:
                        extreme_long_paths += 1
                    avg_path_length += path_len

    # print(rel)
    # make relation dictionary
    with open('/home/wjddn803/PycharmProjects/gct_dual/conceptnet/relation_extract_vocab_update.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            try:
                for i in line[1:]:
                    cn = ast.literal_eval(i)
                    token.append(cn[0])
                    token.append(cn[2])
                    rel.append([cn[1]])
                # print(line)
            except IndexError:
                print('pass')

    # make vocabularies
    # print(token)
    # print(rel)
    token_vocab, token_char_vocab = make_vocab(token, char_level=True)
    lemma_vocab, lemma_char_vocab = make_vocab(lemma, char_level=True)
    conc_vocab, conc_char_vocab = make_vocab(conc, char_level=True)

    num_token = sum(len(x) for x in token)
    rel_vocab = make_vocab(rel)

    print('make vocabularies')
    write_vocab(token_vocab, 'token_vocab')
    write_vocab(token_char_vocab, 'token_char_vocab')
    # write_vocab(lemma_vocab, 'lem_vocab')
    # write_vocab(lemma_char_vocab, 'lem_char_vocab')
    write_vocab(conc_vocab, 'concept_vocab')
    write_vocab(conc_char_vocab, 'concept_char_vocab')
    write_vocab(rel_vocab, 'relation_vocab')
