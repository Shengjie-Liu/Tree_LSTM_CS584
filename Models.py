'''
This file has two models:
1) Bi-Directional LSTM (with, without attention, enhanced version)
2) treeLSTM (with, without attention, enhanced version)

@copyright Lun Li, Shengjie Liu
'''

import dgl
import torch
from torch import nn
import torch.nn.functional as F

# Bi-Direction RNN Model
class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, device, att_dim=None, enhanced=False):
        super().__init__()

        # core rnn
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # attn
        self.att_dim = att_dim
        if att_dim != None:
            self.att_dim = att_dim
            self.W_1 = nn.Linear(hidden_dim * 2, att_dim)
            self.W_2 = nn.Linear(att_dim, 1)

        # enhanced
        self.enhanced = enhanced

        # auxiliary vars
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        self.device = device

    # supports 3 modes
    # 1) att_dim = 0: no attn
    # 2）att_dim != 0, att, att_w = None: attn
    # 3) att_dim != 0, att, att_w != None: attn + benchmark
    def forward(self, text, text_lengths, att=None, att_w=None):

        # produce mask
        mask = (text != self.pad_idx).permute(1, 0)

        # embedding
        embedded = self.dropout(self.embedding(text))

        # pad
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        # roll
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpad
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # three cases
        # if no attn, add vectors up
        attention_weights = None
        att_diff = None
        weighted_vec = torch.sum(output, dim=0)

        if self.att_dim != None:
            # att weights
            tmp = torch.relu(self.W_2(torch.relu(self.W_1(output)))).squeeze(2)
            attention_weights = torch.softmax(tmp.masked_fill(mask.permute(1, 0) == 0, 1e-10).unsqueeze(2), dim=0)
            # weighted vector
            context_vec = attention_weights * output
            weighted_vec = torch.sum(context_vec, dim=0)

            # attention diff
            att_diff = torch.zeros(attention_weights.shape).to(self.device)
            if type(att) == torch.Tensor and type(att_w) == torch.Tensor:
                att_diff = (attention_weights - att.unsqueeze(2)) * att_w.unsqueeze(2)

            att_diff = att_diff.squeeze(2)

        return self.fc(weighted_vec), att_diff, attention_weights

# tree LSTM cell with/out attention
# two modes: additive/multiplicative attention
class TreeLSTMCell(nn.Module):

    def __init__(self, x_size, h_size, attention = ""):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias = False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias = False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)
        # if attention is present
        self.attention = attention
        if attention != "":
            self.key = nn.Linear(h_size, h_size)
            self.query = nn.Linear(h_size, h_size)
            self.value = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {"h": edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        if self.attention != "":
            hs = nodes.mailbox['h']
            key = self.key(hs); query = self.query(hs)
            align = torch.bmm(query, torch.transpose(key, 1, 2))
            a = F.softmax(align, dim=-1)
            if self.attention == "MULT":
                h_mod = torch.bmm(a, self.value(hs))
            elif self.attention == "ADD":
                h_mod = torch.tanh(self.value(torch.bmm(a, hs)))
            else:
                raise Exception("Only supports Multiplicative/Additive Attention")
            h_cat = h_mod.view(nodes.mailbox['h'].size(0), -1)
        else:
            h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = torch.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)

        return {'h': h, 'c': c}

# tree LSTM
class TreeLSTM(nn.Module):

    def __init__(self, num_vocabs, x_size, h_size, num_classes, dropout, att_type):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        self.dropout = nn.Dropout(dropout)
        self.att_type = att_type
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = TreeLSTMCell(x_size, h_size, att_type)

    def forward(self, batch, h, c):
        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h; g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits