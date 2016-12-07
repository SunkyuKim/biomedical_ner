
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import time
import os
from model import Model
from utils import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='res/BioCreative2GM/test/', help='data directory')
    parser.add_argument('--restore', type=str, default='None', help='ckpt file path')
    parser.add_argument('--save_dir', type=str, default='logs/', help='ckpt file path')
    parser.add_argument('--batch_size', type=int, default=128, help='data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='num_epoch')
    parser.add_argument('--rnn_size', type=int, default=100, help='output nodes of rnn')
    parser.add_argument('--class_size', type=int, default=3, help='class size')

    args = parser.parse_args()
    test(args)

def test(args):
    data_loader = DataLoader(args.data_dir, args.save_dir, args.batch_size)

    args.vocab_size = data_loader.vocab_size
    args.seq_length = data_loader.max_sequence_length
    args.embedding = np.random.uniform(-10, 10, [args.vocab_size, args.seq_length])
    args.embedding[0] = np.zeros(args.seq_length)

    x, y = data_loader.full_batch()
    args.batch_size = x.shape[0]

    model = Model(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(args.save_dir, args.restore))

    #tf.initialize_all_variables().run()
    start = time.time()
    feed = {model.input_data: x, model.targets: y}

    prediction, loss = sess.run([model.prediction, model.loss], feed)
    end = time.time()

    print("test_loss = {:.3f}, time = {:.3f}"
          .format(loss, end-start))
    tag(prediction)

def tag(prediction):
    o = []
    for p_s in prediction:
        o.append([list(p).index(max(p)) for p in p_s])
    for oo in o[:10]:
        print(oo[:30])

if __name__ == '__main__':
    main()