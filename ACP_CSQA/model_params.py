from collections import defaultdict

BASE_PARAMS = defaultdict(

    # model_condition
    encoder_type='lm',
    lm_model='bert-base-cased',
    bert_pretrained_file=None,
    prefix='model',

    # bert
    bert_embed_dim = 768,
    bert_max_length=256,

    # train
    epochs=20,
    lr=5e-5,
    adam_epsilon=1e-8,
    warmup_steps=2000,
    ckpt='/home/mnt/gct_dual_model',
    seed=42,
    batch_multiplier=1,
    patience = 50,
    # dropout/unk
    dropout=0.2,
    unk_rate=0.33,


    # IO
    task = 'cqa',
    n_answers=5,
    pretrained_file=None,
    block_size=100,

)

AMR_CN_BERT_PARAMS = BASE_PARAMS.copy()
AMR_CN_BERT_PARAMS.update(
    encoder_type='ACB_dual',

    ##### GTOS #####
    token_vocab = './amr_data/amr_2.0/csqa/token_vocab',
    concept_vocab = './amr_data/amr_2.0/csqa/concept_vocab',
    token_char_vocab = './amr_data/amr_2.0/csqa/token_char_vocab',
    concept_char_vocab = './amr_data/amr_2.0/csqa/concept_char_vocab',
    relation_vocab = './amr_data/amr_2.0/csqa/relation_vocab',

    # concept/token encoders
    concept_char_dim = 32,
    concept_dim = 300,
    max_concept_len = 100,
    snt_layer = 1,

    # char-cnn
    cnn_filters = [3,256],
    char2concept_dim = 128,

    # relation encoder
    rel_dim = 100,
    rnn_hidden_size = 256,
    rnn_num_layers = 2,

    # core architecture
    embed_dim = 512,
    ff_embed_dim = 1024,
    num_heads = 8,
    graph_layers = 4,
    n_lstm_layers = 2,


    ##### conceptnet #####
    conceptnet_path = './conceptnet/relation_extract_vocab_update.txt',
    max_conceptnet_length=300,

    ##### BERT #####
    bert_embed_dim = 768,

    ##### TRAIN #####
    task = 'csqa', ############# 이거 꼭 바꾸기!!!!
    n_answers=5,
    lm_model = 'bert-base-cased',
    # bert_pretrained_file = './pretraining_bert/output',
    bert_pretrained_file = None,
    batch_size = 1,
    lr = 3e-5,
    epochs = 20,
    batch_multiplier = 64,
    feature = 'amr_cn_prune01234_pretrained',
    prefix = '',

    ##### IO #####
    pretrained_file = './glove/glove.840B.300d.txt',
    gpus = [7],

    # CSQA - AMRCN prune ARG0, ARG1
    # No official test set
    train_data='/home/mnt/cn_data/amr_2.0/csqa/new_amr_cn_prune_ARG01/train_pred_cn_extended_real_final.json',
    dev_data='/home/mnt/cn_data/amr_2.0/csqa/new_amr_cn_prune_ARG01/dev_pred_cn_extended_real_final.json',
    test_data='/home/mnt/cn_data/amr_2.0/csqa/new_amr_cn_prune_ARG01/dev_pred_cn_extended_real_final.json',
    train_data_jsonl='./amr_data/amr_2.0/csqa/train_rand_split.jsonl',
    dev_data_jsonl='./amr_data/amr_2.0/csqa/dev_rand_split.jsonl',
    test_data_jsonl='./amr_data/amr_2.0/csqa/dev_rand_split.jsonl',

    ckpt='/home/mnt/wjddn803/gct_dual_model'

)
