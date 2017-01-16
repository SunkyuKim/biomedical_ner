from __future__ import print_function
import codecs
import os
import numpy as np
import pickle

class DataLoader():
    def __init__(self, data_dir, save_dir, batch_size, encoding='utf-8', test=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.encoding = encoding

        vocab_file = os.path.join(save_dir, "vocab_best.pkl")
        self.x, self.y = self.preprocess(data_dir, vocab_file)

        self.create_batches()
        self.reset_batch_pointer()

    def load_embedding_lookup_table(self):
        target_embedding = np.random.random([100,50])
        mutation_embedding = np.random.random([100, 50])

        return target_embedding, mutation_embedding

    def build_vocab(self):
        pubmed_data = os.path.join("res/Pubmed/data.txt")

        vocab_dict = dict()
        vocab_size = 1 # 0 is for 'NONE' word.
        ls = open(pubmed_data).readlines()
        label_set = set()
        for l in ls:
            tokens, label = self.sentence_tokenizer(l)
            for t in tokens:
                if t not in vocab_dict:
                    vocab_dict[t] = vocab_size
                    vocab_size += 1
            for l in label:
                label_set.add(l)
        label_dict = dict()
        for i,t in enumerate(sorted(label_set)):
            temp_label_int = [0]*len(label_set)
            temp_label_int[i] = 1
            label_dict[t] = temp_label_int
        label_dict['PAD'] =  [0]*len(label_set)

        for l in label_dict:
            print(l,label_dict[l])

        return vocab_dict, label_dict

    def sentence_tokenizer(self, l):
        text,label = l.split("\t")
        tokens = text.strip().split()
        text_tokens = []
        cleaner = [',', '.', '(', ')']
        for t in tokens:
            if len(t) < 1:
                continue
            if t[0] in cleaner:
                t = t[1:]
            if len(t) < 1:
                continue
            if t[-1] in cleaner:
                t = t[:-1]
            text_tokens.append(t)
        label_tokens = label.strip().split(",")
        return text_tokens, label_tokens

    def preprocess(self, data_dir, vocab_file):
        pubmed_data = os.path.join(data_dir, "data.txt")

        if not (os.path.exists(vocab_file)):
            vocab_dict, label_dict = self.build_vocab()
        else:
            vocab_dict, label_dict = pickle.load(open(vocab_file))

        vocab_size = len(vocab_dict.keys())
        self.vocab_size = vocab_size

        ls = open(pubmed_data).readlines()

        x_text = []
        y_text = []
        for l in ls:
            tokens, label = self.sentence_tokenizer(l)
            x_text.append(tokens)
            y_text.append(label)

        self.max_sequence_length = max(map(len, x_text))

        """
        make x,y_text -> x,y_int
                    """

        x_int = []
        for x in x_text:
            x_int_per_line = [0] * self.max_sequence_length
            for i,t in enumerate(x):
                x_int_per_line[i] = vocab_dict[t]
            x_int.append(x_int_per_line)
        y_int = []
        for y in y_text:
            y_padding = y + ['PAD'] * (self.max_sequence_length-len(y))
            y_int.append(map(label_dict.get, y_padding))

        x_np = np.array(x_int)
        y_np = np.array(y_int)
        print("x : {}".format(x_np.shape))
        print("y : {}".format(y_np.shape))

        return x_np, y_np


    def create_batches(self):
        self.num_batches = int(self.x.size / (self.batch_size *
                                                   self.max_sequence_length))

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.reshape(self.x, self.x.size)
        ydata = np.reshape(self.y, self.y.size)

        xdata = xdata[:self.num_batches * self.batch_size * self.max_sequence_length]
        ydata = ydata[:self.num_batches * self.batch_size * self.max_sequence_length * 17]

        print(self.num_batches, self.batch_size, self.max_sequence_length)

        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1, 17), self.num_batches, 1)


    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def full_batch(self):
        return self.x, self.y

if __name__ == '__main__':
    DataLoader("res/Pubmed", "save", 128)
