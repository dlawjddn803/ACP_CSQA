import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from encoder import RelationEncoder, TokenEncoder
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from graph_transformer import GraphTransformer
from utils import *


class Reasoning_AMR_CN_DUAL(nn.Module):
    def __init__(self, vocabs,
                 concept_char_dim, concept_dim,
                 cnn_filters, char2concept_dim,
                 rel_dim, rnn_hidden_size, rnn_num_layers,
                 embed_dim, bert_embed_dim, ff_embed_dim, num_heads, dropout,
                 snt_layer,
                 graph_layers,
                 pretrained_file, device, batch_size,
                 model_type, bert_config, bert_model, bert_tokenizer, bert_max_length,
                 n_answers,
                 model,
                 max_conceptnet_length,
                 conceptnet_path):

        super(Reasoning_AMR_CN_DUAL, self).__init__()
        self.vocabs = vocabs
        self.embed_scale = math.sqrt(embed_dim)
        self.concept_encoder = TokenEncoder(vocabs['concept'], vocabs['concept_char'],
                                            concept_char_dim, concept_dim, embed_dim,
                                            cnn_filters, char2concept_dim, dropout, pretrained_file)
        self.relation_encoder = RelationEncoder(vocabs['relation'], rel_dim, embed_dim, rnn_hidden_size, rnn_num_layers,
                                                dropout)
        self.graph_encoder = GraphTransformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.transformer = Transformer(snt_layer, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        self.c_transformer = Transformer(snt_layer, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        self.f_transformer = Transformer(snt_layer, bert_embed_dim+embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)

        self.pretrained_file = pretrained_file
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.max_conceptnet_len = max_conceptnet_length
        self.embed_scale = math.sqrt(embed_dim)

        self.token_position = SinusoidalPositionalEmbedding(embed_dim, device)
        self.concept_depth = nn.Embedding(32, embed_dim)
        self.token_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_mask = SelfAttentionMask(device)
        self.dropout = dropout
        self.conceptnet_path = conceptnet_path

        self.bert_config = bert_config
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_max_length = bert_max_length
        self.answer_len = n_answers
        self.device = device
        self.model_type = model_type
        self.model = model
        self.batch_size = batch_size

        self.classifier = nn.Linear(bert_embed_dim+embed_dim, 1)
        self.loss_fct = CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.concept_depth.weight, 0.)

    def encoder_attn(self, inp):
        with torch.no_grad():
            concept_repr = self.embed_scale * self.concept_encoder(inp['concept'],
                                                                   inp['concept_char'] + self.concept_depth(
                                                                       inp['concept_depth']))
            concept_repr = self.concept_embed_layer_norm(concept_repr)
            concept_mask = torch.eq(inp['concept'], self.vocabs['concept'].padding_idx)

            relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
            relation[0, :] = 0.
            relation = relation[inp['relation']]
            sum_relation = relation.sum(dim=3)
            num_valid_paths = inp['relation'].ne(0).sum(dim=3).clamp_(min=1)
            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation)
            relation = sum_relation / divisor

            attn, attn_weights = self.graph_encoder.get_attn_weights(concept_repr, relation, self_padding_mask=concept_mask)

        return attn

    def encode_cn_step(self, inp, i, train=True):
        cn_concept_input = inp['cn_concept'][i][:,0].unsqueeze(1)
        cn_concept_char_input = inp['cn_concept_char'][i][:,0].unsqueeze(1)
        cn_concept_depth_input = inp['cn_concept_depth'][i][:,0].unsqueeze(1)
        cn_relation_bank_input = inp['cn_relation_bank'][i]
        cn_relation_length_input = inp['cn_relation_length'][i]
        cn_relation_input = inp['cn_relation'][i][:,:,0].unsqueeze(2)

        concept_repr = self.embed_scale * self.concept_encoder(cn_concept_input,
                                                               cn_concept_char_input) + self.concept_depth(
            cn_concept_depth_input)
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_mask = torch.eq(cn_concept_input, self.vocabs['concept'].padding_idx)
        relation = self.relation_encoder(cn_relation_bank_input, cn_relation_length_input)

        if str(train)=='True':
            relation = relation.index_select(0, cn_relation_input.reshape(-1)).view(*cn_relation_input.size(), -1)

        else:
            relation[0, :] = 0. # cn_relation_length x dim
            relation = relation[cn_relation_input]  # i x j x bsz x num x dim

            sum_relation = relation.sum(dim=3)  # i x j x bsz x dim
            num_valid_paths = cn_relation_input.ne(0).sum(dim=3).clamp_(min=1)  # i x j x bsz

            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation) # i x j x bsz x 1
            relation = sum_relation / divisor # i x j x bsz dim
        concept_repr = self.graph_encoder(concept_repr, relation, self_padding_mask=concept_mask)

        return concept_repr

    def encoder_cn_attn(self, inp, i):
        with torch.no_grad():
            cn_concept_input = inp['cn_concept'][i][:, 0].unsqueeze(1)
            cn_concept_char_input = inp['cn_concept_char'][i][:, 0].unsqueeze(1)
            cn_concept_depth_input = inp['cn_concept_depth'][i][:, 0].unsqueeze(1)

            cn_relation_bank_input = inp['cn_relation_bank'][i]
            cn_relation_length_input = inp['cn_relation_length'][i]
            cn_relation_input = inp['cn_relation'][i][:, :, 0].unsqueeze(2)

            concept_repr = self.embed_scale * self.concept_encoder(cn_concept_input,
                                                                   cn_concept_char_input) + self.concept_depth(
                cn_concept_depth_input)
            concept_repr = self.concept_embed_layer_norm(concept_repr)
            concept_mask = torch.eq(cn_concept_input, self.vocabs['concept'].padding_idx)

            relation = self.relation_encoder(cn_relation_bank_input, cn_relation_length_input) # [211, 512]

            relation[0, :] = 0.  # cn_relation_length x dim
            relation = relation[cn_relation_input]  # i x j x bsz x num x dim
            sum_relation = relation.sum(dim=3)  # i x j x bsz x dim
            num_valid_paths = cn_relation_input.ne(0).sum(dim=3).clamp_(min=1)  # i x j x bsz

            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation)  # i x j x bsz x 1
            relation = sum_relation / divisor  # i x j x bsz x dim

            concept_repr = self.graph_encoder(concept_repr, relation, self_padding_mask=concept_mask)

            attn = self.graph_encoder.get_attn_weights(concept_repr, relation, self_padding_mask=concept_mask)
        return attn

    def convert_batch_to_bert_features(self,
                                       data,
                                       max_seq_length,
                                       tokenizer,
                                       cls_token_at_end=False,
                                       cls_token='[CLS]',
                                       cls_token_segment_id=1,
                                       sep_token='[SEP]',
                                       sequence_a_segment_id=0,
                                       sequence_b_segment_id=1,
                                       sep_token_extra=False,
                                       pad_token_segment_id=0,
                                       pad_on_left=False,
                                       pad_token=0,
                                       mask_padding_with_zero=True):
        features = []
        questions = [" ".join(x for x in sent) for sent in data['token_data']]
        answers = data['answers']

        choices_features = []
        for i, text in enumerate(questions):
            question = text

            for j, ans in enumerate(answers[i]):
                answer = answers[i][j]

                token_a = tokenizer.tokenize(question)
                token_b = tokenizer.tokenize(answer)

                tokens = token_a + [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)

                if token_b:
                    tokens += token_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(token_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if padding_length > 0:
                    if pad_on_left:
                        input_ids = ([pad_token] * padding_length) + input_ids
                        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                    else:
                        input_ids = input_ids + ([pad_token] * padding_length)
                        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                else:
                    input_ids = input_ids[:max_seq_length]
                    input_mask = input_ids[:max_seq_length]
                    segment_ids = segment_ids[:max_seq_length]

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((input_ids, input_mask, segment_ids))

            features.append(choices_features)
            choices_features = []

        return features


    def prepare_bert_input(self, data, tokenizer):
        move_to_cuda(data, self.device)

        features = self.convert_batch_to_bert_features(data = data,
                                                  max_seq_length=self.bert_max_length,
                                                  tokenizer=tokenizer,
                                                  )

        move_to_cuda(features, self.device)

        all_input_ids = torch.tensor([f[0] for feature in features for f in feature], dtype=torch.long).view(self.batch_size, -1, self.bert_max_length).to(self.device)
        all_input_mask = torch.tensor([f[1] for feature in features for f in feature], dtype=torch.long).view(self.batch_size, -1, self.bert_max_length).to(self.device)
        all_segment_ids = torch.tensor([f[2] for feature in features for f in feature], dtype=torch.long).view(self.batch_size, -1, self.bert_max_length).to(self.device)

        return all_input_ids, all_input_mask, all_segment_ids


    def prepare_graph_state(self, graph_state, ans_len, concept_dim):
        tot_initial = torch.tensor(1).to(self.device)

        j = 0
        while j < (1*ans_len)-1:
            initial = graph_state[0][j].view(1, -1).to(self.device)
            for i in graph_state[1:]:  # i = [5 x 512] x 7개
                com_tensor = i[j + 1].view(1, -1).to(self.device)

                initial = torch.cat([initial, com_tensor], dim=0)
            if j == 0:
                tot_initial = initial.view(1, -1, concept_dim)
            j += 1
            initial = initial.view(1, -1, concept_dim)
            tot_initial = torch.cat([tot_initial, initial], dim=0)
        return tot_initial

    def forward(self, data, train):
        answer_len = self.answer_len
        tot_concept_reprs = []
        for i in range(self.batch_size):
            ## AMR-GTOS
            concept_repr = self.encode_cn_step(data, i, train=train)  # concept_seq_len x 1 x concept_embed_size
            concept_repr = self.transformer(concept_repr, kv=None)  # res = concept_seq_len x bsz x concept_embed_size


            if concept_repr.size()[1] == 1:
                concept_repr = concept_repr.squeeze().unsqueeze(0).mean(1).unsqueeze(1)

            else:
                concept_repr = self.prepare_graph_state(concept_repr, concept_repr.size()[1], self.embed_dim).mean(
                    1).unsqueeze(1)  # re = bsz x 1 x concept_embed_size

            concept_repr = concept_repr.repeat(1, answer_len, 1)  # re = 1 x 5 x concept_embed_size
            tot_concept_repr = self.c_transformer(concept_repr, kv=None)
            tot_concept_reprs.append(tot_concept_repr)

        tot_concept_reprs = torch.squeeze(torch.stack(tot_concept_reprs), 1)

        ids = data['id']
        labels = data['answer_key']
        labels = torch.tensor(labels).to(self.device)

        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        ### prepare bert input
        # 3 x 5 x 128
        all_input_ids, all_input_mask, all_segment_ids = self.prepare_bert_input(data,
                                                                                 self.bert_tokenizer)
        logits = self.bert_model(
            input_ids=all_input_ids,
            attention_mask=all_input_mask,
            token_type_ids=all_segment_ids if self.model_type in ['bert-base-cased', 'xlnet-base-cased'] else None,
            labels=labels,
            n_answers=answer_len
        )
        bsz = len(ids)

        final_logits = torch.cat([logits, tot_concept_reprs], 2)
        final_logits = self.f_transformer(final_logits, kv=None)
        final_logits = self.classifier(final_logits).squeeze().view(bsz, -1)

        return final_logits, labels, ids
