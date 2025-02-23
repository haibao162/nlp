from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words # 第一个词是2，第二个词是3  {'你': 2, '好': 3}
            self.word2count[word] = 1
            self.index2word[self.n_words] = word # 把词放在index2word里面: {0: "SOS", 1: "EOS", 2: '你', 3:'好'}
            self.n_words += 1
        else:
            self.word2count[word] += 1 # 词的次数需要记录。


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
    # 为了快速展示结果，我们减少一下使用的训练数据
    # pairs = pairs[:10000]
 
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    # print(len(input_lang), len(output_lang))
    print(input_lang.name, input_lang.n_words) #4972
    print(output_lang.name, output_lang.n_words)#3179
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        # print(input_size, 'input_size', 'hidden_size')
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        # torch.Size([1])
        # print(self.embedding(input).shape, 'self.embedding(input).shape')
        # torch.Size([1, 256])
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        # out, _ = self.gru(x, h0)里的的初始值h0
        return torch.zeros(1,1, self.hidden_size, device=device)
    


# The Decoder
# -----------
# 
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
# 
# 
# 
 
# Simple Decoder
# ^^^^^^^^^^^^^^
# 
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
# 
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # output_size * hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size) # batch_size * sentence_length * hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print(input.shape, 'input.shape')
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size, device=device)
    

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        # decoder_output, decoder_hidden, decoder_attention = decoder(
        #         decoder_input, decoder_hidden, encoder_outputs)

        # print(input.shape, 'input input.shape')
        # print(hidden.shape, 'hidden hidden.shape')
        # print(encoder_outputs.shape, 'encoder_outputs encoder_outputs.shape')
        # torch.Size([1]) input input.shape
        # torch.Size([1, 1, 256]) hidden hidden.shape
        # torch.Size([10, 256]) encoder_outputs encoder_outputs.shape
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)

        # input真实值和上一次计算的hidden合并
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) #1,max_length
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) #unsqueeze外层加上一维，bmm计算三维的矩阵相乘

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Training
# ========
# 
# Preparing Training Data
# -----------------------
# 
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
# 

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    # 例如：['je ne porte aucun sous vetement .', 'i am not wearing any underwear .']
    # print(input_lang.word2index, 'input_lang') # 4971
    # print(output_lang.word2index, 'output_lang') # 3178

    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    # print(input_tensor, 'input_tensor')
    # print(target_tensor, 'target_tensor')
#     tensor([[171],
#         [272],
#         [467],
#         [  5],
#         [  1]]) input_tensor
# tensor([[ 96],
#         [ 47],
#         [265],
#         [  4],
#         [  1]]) target_tensor
    return (input_tensor, target_tensor)
                        
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0) # 输入的长度
    target_length = target_tensor.size(0) # 输出的长度
    # target_tensor例如：'i am not wearing any underwear .'

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        # encoder进去的是一个单词
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden) # 获取encoder层结果：1 * 1 * 256
        encoder_outputs[ei] = encoder_output[0,0] # 输出的256个维度的结果: torch.Size([256])

    decoder_input = torch.tensor([[SOS_token]], device=device) # decoder层一开始是SOS_TOKEN

    decoder_hidden = encoder_hidden # gru层和RNN一样，第二个参数是输出的结果，和第一个参数[:,-1,:]的值相等

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    use_teacher_forcing = True

    if use_teacher_forcing:
        for di in range(target_length):
            # decoder_hidden是encoder层计算得到的结果，也就是最后一个字的输出，decoder_input是decoder层的输入，teacher forcing模式下从真实值target_tensoe里获取
            # encoder_outputs取每个输入的字的一部分，由于是一个字，[0,0]就是每个字过encoder后的输出，encoder_outputs就代表输入的句子
            # decoder_hidden来自输入，代表输出，encoder_outputs来自输入，代表输入的那句话。decoder_input来自输出
            # decoder_hidden第一次是取encoder的输出，后面每次都从上一次取
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # AttnDecoderRNN
            # CrossEntropyLoss=NLLLoss+LogSoftmax
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] # 按照真实值一个个输出，加到下一次decoder里
    
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs)) 

def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
 
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
                        
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)


        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0,0]
        
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1) # 获取最大值的值，索引
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        
        return decoded_words, decoder_attentions[:di+1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
 
trainIters(encoder1, attn_decoder1, 1000, print_every=100)


evaluateRandomly(encoder1, attn_decoder1)

# test = tensorsFromPair(random.choice(pairs))
