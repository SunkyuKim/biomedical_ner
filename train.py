
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import time
import os
from model import Model
# from utils import DataLoader
from utils_best import DataLoader
import procname

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='res/BioCreative2GM/train/', help='data directory')
    parser.add_argument('--data_dir', type=str, default='res/Pubmed/train/', help='data directory')
    parser.add_argument('--restore', type=str, default=None, help='ckpt file name')
    parser.add_argument('--save_dir', type=str, default='logs/pubmed/', help='ckpt file path')
    parser.add_argument('--batch_size', type=int, default=1000, help='data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='num_epoch')
    parser.add_argument('--rnn_size', type=int, default=100, help='output nodes of rnn')
    # parser.add_argument('--class_size', type=int, default=3, help='class size')
    parser.add_argument('--class_size', type=int, default=17, help='class size')
    parser.add_argument('--save_every', type=int, default=100, help='save per iteration')
    parser.add_argument('--exp_code', type=str, default=None, help='Experiment code')

    args = parser.parse_args()
    if args.exp_code == None:
        print("YOU SHOULD INPUT EXP_CODE")
        exit()
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    if os.path.isdir(args.save_dir):
        print("INPUT EXP_CODE EXISTS ALREADY")
        exit()
    else:
        os.mkdir(args.save_dir)

    procname.setprocname("NER_Pubmed_TRAIN")
    print(args)
    train(args)

def train(args):
    data_loader = DataLoader(args.data_dir, args.save_dir, args.batch_size)

    args.vocab_size = data_loader.vocab_size
    args.seq_length = data_loader.max_sequence_length
    args.embedding = np.random.uniform(-10, 10, [args.vocab_size, args.seq_length])
    args.embedding[0] = np.zeros(args.seq_length)

    # args.target_embedding_np, args.mutation_embedding_np \
    #     = data_loader.load_embedding_lookup_table()

    if args.restore is not None:
        assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from

    model = Model(args)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(os.path.join(args.save_dir, 'train'), sess.graph)

        if args.restore is not None:
            pass
            # saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            # sess.run(tf.assign(model.tf_learning_rate, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}

                train_loss, _, summary = sess.run([model.loss, model.train_op, model.merged], feed)
                end = time.time()
                train_writer.add_summary(summary, global_step=e * data_loader.num_batches + b)

                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))

                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
        train_writer.close()

if __name__ == '__main__':
    main()
