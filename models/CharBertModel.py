import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig, BertModel, BertEmbeddings, BertPooler, BertEncoder, BertPreTrainedModel

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# model
class CharEmbedding(BertEmbeddings):
    def __init__(self, config, g_seq, s_seq):
        super(CharEmbedding, self).__init__(config)
        assert g_seq >= 0
        self.graph_sequence = g_seq
        self.sent_sequence=s_seq

    def forward(self, character_embeddings, input_ids, token_type_ids=None, attention_mask=None):

        words_embeddings = self.word_embeddings(input_ids)
       
        if character_embeddings !=None:
            words_embeddings=torch.cat([character_embeddings, words_embeddings], dim=1)[:, :self.sent_sequence, :]
        else:
            words_embeddings = self.word_embeddings(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class CharBertModel(BertModel):
    def __init__(self, config, g_dim, w_dim, num_labels, output_attentions=False,
                 keep_multihead_output=False):
        super(CharBertModel, self).__init__(config, output_attentions, keep_multihead_output)
        self.config = config
        self.embeddings=CharEmbedding(config, g_dim, w_dim)
        self.encoder = BertEncoder(config, output_attentions=output_attentions,
                                   keep_multihead_output=keep_multihead_output)
        self.pooler = BertPooler(config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.output_attentions = output_attentions

        self.apply(self.init_bert_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, character_embedding, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=False, head_mask=None):

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if character_embedding==None:
            embedding_output = self.embeddings(None, input_ids, token_type_ids, attention_mask)
        else:
            embedding_output=self.embeddings(character_embedding, input_ids, token_type_ids, attention_mask)
        

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if self.output_attentions:
            output_all_encoded_layers = True
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits

        return logits