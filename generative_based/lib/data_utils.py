import sys, os, re, gzip, collections
import tensorflow as tf 


_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_POS = b"_POS"
_NEG = b"_NEG"

_START_VOCAB = [_GO, _EOS, _UNK, _POS, _NEG]

GO_ID = 0
EOS_ID = 1
UNK_ID = 2
POS_ID = 3
NEG_ID = 4

def set_file(workspace, dataspace):
    pos_path = os.path.join(dataspace, 'pos_file.txt')
    neg_path = os.path.join(dataspace, 'neg_file.txt')
    for f in [workspace]:
        if not os.path.exists(f): os.makedirs(f)
    return pos_path, neg_path

def creat_vocabulary(vocab_path, pos_path, neg_path, vocab_size):
    if not os.path.exists(vocab_path):
        text = []
        with open(pos_path, 'r') as fin:
            for line in fin:
                text.extend(line.strip().split())
        with open(neg_path, 'r') as fin:
            for line in fin:
                text.extend(line.strip().split())
        count = collections.Counter(text).most_common(vocab_size-5)
        dict_list = [w for w, _ in count]
        dict_list = _START_VOCAB + dict_list
        with open(vocab_path, 'w') as fout:
            for v, k in enumerate(dict_list):
                fout.write(str(k)+' '+str(v)+'\n')

def initialize_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        print ("%s not found!!" %vocab_path)
    else:
        vocab, rev_vocab = {}, {}
        with open(vocab_path, 'r') as fin:
            for line in fin:
                line = line.strip().split()
                vocab[str(line[0])] = int(line[1])
                rev_vocab[int(line[1])] = str(line[0])
        return vocab, rev_vocab

def sentence_to_id(sentence, vocab):
    line = sentence.split()
    return [str(vocab.get(w, UNK_ID)) for w in line]

def data_to_id(data_id_path, data_path, vocab_path):
    if not os.path.exists(data_id_path):
        print ('Tokenizing data in %s' % data_path)
        vocab, _ = initialize_vocab(vocab_path)
        with open(data_path, 'r') as fin:
            with open(data_id_path, 'w') as fout:
                counter = 0
                for line in fin:
                    counter += 1
                    if counter%100000 == 0:
                        print ("\r--tokenizing line %d" %counter, end='')
                    token_line = sentence_to_id(line.strip(), vocab)
                    fout.write(' '.join(token_line) + '\n')
                print ()

def prepare_diologue(workspace, dataspace, vocab_size):
    pos_path, neg_path = set_file(workspace, dataspace)
    pos_id_path = os.path.join(dataspace, ('pos_id_%d.in' % vocab_size)) 
    neg_id_path = os.path.join(dataspace, ('neg_id_%d.in' % vocab_size) )
    vocab_path  = os.path.join(dataspace, ('vocab_%d.in' % vocab_size) )
    creat_vocabulary(vocab_path, pos_path, neg_path, vocab_size)
    data_to_id(pos_id_path, pos_path, vocab_path)
    data_to_id(neg_id_path, neg_path, vocab_path)
    return pos_id_path, neg_id_path, vocab_path

def prepare_test_data(workspace, dataspace, vocab_size):
    data_path = os.path.join(dataspace, 'seq2seq.txt')
    data_id_path = os.path.join(dataspace, ('seq2seq_id_%d.in' % vocab_size)) 
    vocab_path  = os.path.join(dataspace, ('vocab_%d.in' % vocab_size) )
    data_to_id(data_id_path, data_path, vocab_path)
    return data_id_path, vocab_path

def read_data(data_path, sentiment):
    with open(data_path, 'r') as fin:
        source= fin.readline()
        counter = 0
        data_set = []
        while source:
            counter += 1
            if counter % 100000 == 0:
                print ("\r--reading data line %d" %counter, end='')
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            data_set.append((source_ids, sentiment))
            source = fin.readline()
        print ()
    return data_set