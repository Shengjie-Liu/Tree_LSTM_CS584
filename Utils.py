'''
The utilities caontins:
1) Basic Utils;
2) RNN Helpers;
3) Tree Helpers;
4) Results Analyzers;
'''

import random
import numpy as np
import pandas as pd
from collections import OrderedDict

import spacy
import torch
from torchtext import data
import torch.nn.functional as F

import dgl
import networkx as nx
from dgl.data import SSTBatch
from nltk.corpus.reader import BracketParseCorpusReader

# language setting
MAX_LEN_ATT = 6
ATT_W1 = 1.
ATT_W2 = 2.
PAD_WORD = -1
UNK_WORD = -1
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
DEVICE = torch.device("cuda")

#######################################################################
#                             Gen Utils                               #
#######################################################################

# tokenizer
def tokenizer(s):
    return [w.text.strip() for w in nlp(s)]

# load training, test data
def tabularData(train_file, test_file, device, batch_size, glove_dim = 100, sr = 0.9, seed = 1, extra = True):

    label = data.LabelField()
    text = data.Field(tokenize = tokenizer, include_lengths = True, init_token = '<sos>', eos_token = '<eos>')
    test_fields = [ ('text', text), ('sentiment', label)]
    if extra: train_fields = [('text', text), ('selected_text', text), ('sentiment', label)]
    else: train_fields = [('text', text), ('sentiment', label)]
    # prepare train/dev/test data
    train_data = data.TabularDataset(path = train_file + '.csv', format = 'csv', fields = train_fields, skip_header = True)
    test = data.TabularDataset(path = test_file + '.csv', format = 'csv', fields = test_fields, skip_header = True)
    train, valid = train_data.split(random_state = random.seed(seed), split_ratio = sr)
    # build vocab
    text.build_vocab(train, vectors = "glove.6B." + str(glove_dim) + "d", unk_init = torch.Tensor.normal_)
    label.build_vocab(train)
    tr_iter, v_iter, ts_iter = data.BucketIterator.splits((train, valid, test),
                                    sort_key = lambda x: len(x.text), batch_size = batch_size,
                                    sort_within_batch = True, device = device)
    
    return tr_iter, v_iter, ts_iter, text

#######################################################################
#                             RNN LSTM                                #
#######################################################################

# save trained model word embedding
def saveWordVecFromModel(itos, word_vecs):
    word_emb = dict()
    for i in range(len(itos)): word_emb[itos[i]] = word_vecs[i].tolist()
    return word_emb

# update model with word embedding
def updateWordVecToModel(itos, org_vectors, word_emb, device):
    for i in range(len(itos)):
        if itos[i] in word_emb.keys():
            org_vectors[i] = torch.FloatTensor(word_emb[itos[i]]).to(device)
    return torch.FloatTensor(org_vectors)

# accuracy(percentage)
def accuracy(prediction, y):
    max_preds = prediction.argmax(dim = 1, keepdim = True)
    label = max_preds.squeeze(1).eq(y)
    return label.sum() / torch.FloatTensor([y.shape[0]])

# make attention benchmark (if qualified)
def makeAttentionBenchmark(text, selected_text, device, ATT_W1 = 1, ATT_W2 = 5, EPSILON = 0.5):

    # att, att_w
    b_len, b_size = text.shape
    att_w_1, att_w_2, scaling = ATT_W1, ATT_W2, 1. / b_len
    att, att_w = torch.zeros(b_size, b_len).to(device), torch.zeros(b_size, b_len).to(device)
    for idx in range(b_size):
        text_len = len(text[:, idx])
        sel_len = (selected_text[:, idx] != 1).sum() - 2
        if sel_len <= EPSILON * text_len:
            found, pos = False, []
            att[idx] = torch.tensor([0.] * b_len)
            att_w[idx] = torch.tensor([att_w_1] * b_len)
            # find those positions
            for i in range(1, sel_len + 1):
                tmp = (text[:, idx] == selected_text[:, idx][i]).nonzero()
                if len(tmp) != 0:
                    found = True
                    pos.append(int(tmp[0]))
            if found:
                # fill those positions
                for each in pos:
                    att[idx][each] = 1. / len(pos)
                    att_w[idx][each] = att_w_2
            else:
                att[idx], att_w[idx] = torch.tensor([0] * b_len), torch.tensor([0] * b_len)
        else:
            att[idx], att_w[idx] = torch.tensor([0] * b_len), torch.tensor([0] * b_len)

    return att.permute(1, 0), att_w.permute(1, 0)

# training function - LSTM
def training_LSTM(model, train_iter, optimizer, criterion, device, lambda_ = 0.25, power = 2):

    # initialize vars
    epoch_loss = 0
    epoch_acc = 0
    epoch_att_loss = 0
    enhanced = model.enhanced
    att_on = False if model.att_dim == None else True

    # set status to train to turn on gradient calculations
    model.train()

    # loop through batches
    for batch in train_iter:

        # reset grad
        optimizer.zero_grad()
        # fetch batch dta
        text, text_lengths = batch.text
        _, b_size = text.shape

        # core part
        predictions, att_diff_w, att_loss = None, None, 0
        if not att_on:
            # this is the case working with selected_text only
            selected_text, selected_text_lengths = batch.selected_text
            predictions, _, _ = model(selected_text, selected_text_lengths)
        else:
            if enhanced:
                selected_text, selected_text_lengths = batch.selected_text
                att, att_w = makeAttentionBenchmark(text, selected_text, device)
                predictions, att_diff_w, _ = model(text, text_lengths, att, att_w)
                # get att loss
                if power == 1: att_loss = abs(att_diff_w).sum() / b_size
                else: att_loss = (att_diff_w ** 2).sum() / b_size
                # recording att loss
                epoch_att_loss += att_loss.item() * lambda_
            else:
                predictions, _, _ = model(text, text_lengths)

        # get pred loss
        predictions = predictions.squeeze(1)
        pre_loss = criterion(predictions, batch.sentiment)
        # aggregate
        loss = lambda_ * att_loss + pre_loss
        # backpropogate
        loss.backward()
        # advance state
        optimizer.step()
        # reporting
        epoch_loss += pre_loss.item()
        epoch_acc += accuracy(predictions, batch.sentiment).item()

    return epoch_loss / len(train_iter), epoch_acc / len(train_iter), epoch_att_loss / len(train_iter)

# evaluation - LSTM
def evaluation_LSTM(model, test_iterator, criterion):

    # init
    epoch_loss = 0
    epoch_acc = 0
    att_on = False if model.att_dim == None else True

    # stop calculating gradients
    model.eval()

    with torch.no_grad():
        # main loop
        for batch in test_iterator:
            text, text_lengths = batch.text
            if not att_on:
                selected_text, selected_text_lengths = batch.selected_text
                predictions, _, _ = model(selected_text, selected_text_lengths)
            else:
                predictions, _, _ = model(text, text_lengths)
            loss = criterion(predictions.squeeze(1), batch.sentiment)
            acc = accuracy(predictions, batch.sentiment)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(test_iterator), epoch_acc / len(test_iterator)

# Function to extract attention
def extract_att_weights(model, sentence, vocab, device):
    sen_index = [vocab.stoi[each] for each in sentence]  # convert to index
    sen = torch.LongTensor(sen_index).unsqueeze(1).to(device)  # convert to tensor
    sen_length = torch.LongTensor([len(sen_index)]).to(device)  # length to tensor
    predictions, _, alpha = model(sen, sen_length)
    pre_class = predictions.squeeze(0).squeeze(0).argmax()
    return int(pre_class), alpha.squeeze(1).squeeze(1).tolist()

# helper class
class wordVal(object):

    def __init__(self, word, val):
        self.word = word
        self.val = val

    def __str__(self):
        return self.word

# coloring
def color_word(s):
    r = 255 - int(s.val * 255)
    color = '#%02x%02x%02x' % (255, r, r)
    return 'background-color: %s' % color

# visualize attention
def show_attention(sen, att):
    pairs = [wordVal(w, v) for w,v in zip(sen, att)]
    df = pd.DataFrame(pairs).transpose()
    return df.style.applymap(color_word)

#######################################################################
#                             TREE LSTM                               #
#######################################################################

# define tree building function
def build_tree(root, vocab):

    g = nx.DiGraph()

    def _rec_build(nid, node):
        for child in node:
            cid = g.number_of_nodes()
            if isinstance(child[0], str) or isinstance(child[0], bytes):
                # leaf node
                word = vocab.get(child[0].lower(), UNK_WORD)
                if word != -1:
                    g.add_node(cid, x = word, y = int(child.label()), mask = 1)
                else:
                    g.add_node(cid, x = word, y = int(child.label()), mask = 0)
            else:
                g.add_node(cid, x = PAD_WORD, y = int(child.label()),mask = 0)
                _rec_build(cid, child)
            g.add_edge(cid, nid)

    # add root
    g.add_node(0, x = PAD_WORD, y = int(root.label()), mask = 0)
    _rec_build(0, root)
    ret = dgl.DGLGraph()
    ret.from_networkx(g, node_attrs= ['x', 'y', 'mask'])
    return ret

# convert text tree to DGL object
def text2DGL(source_file, vocab_file, embed_file, word_dim):

    # vocab(stoi): {word : index}
    vocab = OrderedDict()
    with open(vocab_file, encoding='utf-8') as vf:
        for line in vf.readlines():
            line = line.strip()
            vocab[line] = len(vocab)

    # enrich word embedding
    embedding = np.random.random((len(vocab), word_dim))
    with open(embed_file, 'r', encoding='utf-8') as pf:
        for line in pf.readlines():
            sp = line.split(' ')
            if sp[0].lower() in vocab:
                embedding[vocab[sp[0].lower()]] = np.array([float(x) for x in sp[1:]])

    # build dgl from file
    files = [source_file]
    corpus = BracketParseCorpusReader('{}'.format(""), files)
    sents = corpus.parsed_sents(files[0])
    trees = [build_tree(sent, vocab) for sent in sents]
    return trees, embedding, vocab

# batcher
def batcher(dev):

    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        mask=batch_trees.ndata['mask'].to(DEVICE),
                        wordid=batch_trees.ndata['x'].to(DEVICE),
                        label=batch_trees.ndata['y'].to(DEVICE))

    return batcher_dev

# trianing/evaluation functions
def evaluation_treeLSTM(model, test_loader, device, enhanced = False):

  acc_vec, loss_vec = [], []
  model.eval()

  with torch.no_grad():

    for step, batch in enumerate(test_loader):
      g = batch.graph
      n = g.number_of_nodes()
      h = torch.zeros((n, model.embedding.embedding_dim)).to(device)
      c = torch.zeros((n, model.embedding.embedding_dim)).to(device)
      logits = model(batch, h, c)
      logp = F.log_softmax(logits, 1)
      # only code
      root_indices = [0] + list(np.array(g.batch_num_nodes).cumsum())[:-1]
      w = torch.FloatTensor([1. if i in root_indices else 0 for i in range(len(batch.label))]).to(device)
      w_lambda = w
      if enhanced:
        w_lambda = torch.FloatTensor([1. if i in root_indices else 0.1 for i in range(len(batch.label))]).to(device)
      loss = (F.nll_loss(logp, batch.label, reduction='none') * w_lambda).sum()
      # ------------------------------------------------------------------------------------------------------
      pred = torch.argmax(logits, 1)
      noise = len(w) - w.long().sum()
      acc = float(torch.sum(torch.eq(batch.label * w.long(), pred * w.long())) - noise) / len(root_indices)
      loss_vec.append(loss.item()); acc_vec.append(acc)

  return np.mean(loss_vec), np.mean(acc_vec)

def training_treeLSTM(model, train_loader, optimizer, device, enhanced = False):

  acc_vec, loss_vec = [], []
  model.train()

  for step, batch in enumerate(train_loader):

    g = batch.graph
    n = g.number_of_nodes()
    h = torch.zeros((n, model.embedding.embedding_dim)).to(device)
    c = torch.zeros((n, model.embedding.embedding_dim)).to(device)
    logits = model(batch, h, c)
    logp = F.log_softmax(logits, 1)
    # only code
    root_indices = [0] + list(np.array(g.batch_num_nodes).cumsum())[:-1]
    w = torch.FloatTensor([1. if i in root_indices else 0 for i in range(len(batch.label))]).to(device)
    w_lambda = w
    if enhanced:
        w_lambda = torch.FloatTensor([1. if i in root_indices else 0.1 for i in range(len(batch.label))]).to(device)
    loss = (F.nll_loss(logp, batch.label, reduction='none') * w_lambda).sum()
    # ------------------------------------------------------------------------------------------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = torch.argmax(logits, 1)
    noise = len(w) - w.long().sum()
    acc = float(torch.sum(torch.eq(batch.label * w.long(), pred * w.long())) - noise) / len(root_indices)
    loss_vec.append(loss.item()); acc_vec.append(acc)

  return np.mean(loss_vec), np.mean(acc_vec)
