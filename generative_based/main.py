import sys, os, argparse
import tensorflow as tf 
import numpy as np 

from model import sentiment_trans

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-train', action='store_true', help='train...')
	parser.add_argument('-chat', action='store_true', help='chat...')
	parser.add_argument('-test', action='store_true', help='test...')
	parser.add_argument('-emb_size' , default=100, help='embedding_dim...')
	parser.add_argument('-max_length' , default=20, help='max_length...')
	parser.add_argument('-latent_dim' , default=500, help='num_cell...')
	parser.add_argument('-batch_size' , default=8, help='batch_size...')
	parser.add_argument('-vocab_size' , default=15000, help='vocab_size...')
	parser.add_argument('-dim' ,type=int, default=256, help='dim...')
	parser.add_argument('-checkpoint_step' , default=100, help='checkpoint_step...')
	parser.add_argument('-model' ,type=str , default='save', help='model_dir...')
	parser.add_argument('-attention', action='store_true', help='using attention model...')
	parser.add_argument('-pretrain', action='store_true', help='pretrain model...')
	parser.add_argument('-data_path',type=str, default='op')
	args = parser.parse_args()
	args.works_dir = 'works/' + args.model
	return args


def run(args):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	# sess = tf.Session()
	print ('---------building graph---------')
	model = sentiment_trans(args, sess)
	if args.train:
		model.train()
	elif args.chat:
		model.chat()
	elif args.test:
		model.test()

if __name__ == '__main__':
	args = parse()
	run(args)

