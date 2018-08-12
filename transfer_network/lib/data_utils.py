import sys, os, re, gzip, collections
import tensorflow as tf 

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def set_file(workspace):
	data_dir = "%s/%s" %(workspace, 'data')
	save_dir = "%s/%s" %(workspace, 'save')
	cyc_dir = "%s/%s" %(workspace, 'trans_save')
	class_dir = "%s/%s" %(workspace, 'class_save')
	train_path = "%s/chat.in" % data_dir
	dev_path = "%s/chat_test.in" % data_dir
	for f in [data_dir, save_dir, cyc_dir, class_dir]:
		if not os.path.exists(f): os.mkdir(f)
	if not os.path.exists(train_path):
		n = 0
		data_file = open("%s/chat.txt" % workspace, 'r')
		train_file = open(train_path, 'w')
		dev_file = open(dev_path, 'w')
		for line in data_file:
			train_file.write(line)
			if n < 10000:
				dev_file.write(line)
				n += 1
	return data_dir, train_path, dev_path

def creat_vocabulary(vocab_path, train_path, vocab_size):
	if not os.path.exists(vocab_path):
		text = []
		with open(train_path, 'r') as fin:
			for line in fin:
				text.extend(line.strip().split())
		count = collections.Counter(text).most_common(vocab_size-4)
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


def prepare_diologue(workspace, vocab_size):
	data_dir, train_path, dev_path = set_file(workspace)
	train_id_path = data_dir + '/chat_id.in'
	dev_id_path = data_dir + '/chat_test_id.in'
	vocab_path = data_dir + ('/vocab%d.in' % vocab_size)
	creat_vocabulary(vocab_path, train_path, vocab_size)
	data_to_id(train_id_path, train_path, vocab_path)
	data_to_id(dev_id_path, dev_path, vocab_path)
	return train_id_path, dev_id_path, vocab_path

def read_data(data_path):
	with open(data_path, 'r') as fin:
		source, target = fin.readline(), fin.readline()
		counter = 0
		data_set = []
		while source and target:
			counter += 1
			if counter % 100000 == 0:
				print ("\r--reading data line %d" %counter, end='')
				sys.stdout.flush()
			source_ids = [int(x) for x in source.split()]
			target_ids = [int(x) for x in target.split()]
			target_ids.append(EOS_ID)
			data_set.append([source_ids, target_ids])
			source, target = fin.readline(), fin.readline()
		print ()
	return data_set

def prepare_cyc_diologue(workspace, vocab_size):
	_, _, _ = set_file(workspace)
	pos_file = ('%s/pos_file.txt' %workspace)
	neg_file = ('%s/neg_file.txt' %workspace)
	data_path = ('%s/data/' %workspace)
	vocab_path = data_path + ('vocab%d.in' % vocab_size)
	pos_id_path = data_path + 'pos_id.in'
	neg_id_path = data_path + 'neg_id.in'
	data_to_id(pos_id_path, pos_file, vocab_path)
	data_to_id(neg_id_path, neg_file, vocab_path)
	return pos_id_path, neg_id_path, vocab_path

def read_cyc_data(data_path):
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
			data_set.append(source_ids)
			source = fin.readline()
		print ()
	return data_set