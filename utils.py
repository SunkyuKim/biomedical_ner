
import codecs
import os
import numpy as np
import cPickle

class DataLoader():
    def __init__(self, data_dir, save_dir, batch_size, encoding='utf-8', test=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.encoding = encoding

        x_file = os.path.join(data_dir, 'train.in')
        y_file = os.path.join(data_dir, 'GENE.eval')
        vocab_file = os.path.join(save_dir, "vocab.pkl")
        self.x, self.y = self.preprocess(data_dir, vocab_file)

        self.create_batches()
        self.reset_batch_pointer()

    def load_embedding_lookup_table(self):
        target_embedding = np.random.random([100,50])
        mutation_embedding = np.random.random([100, 50])

        return target_embedding, mutation_embedding

    def preprocess(self, data_dir, vocab_file):
        x_text = os.path.join(data_dir, 'train.in')
        y_text = os.path.join(data_dir, 'GENE.eval')

        # make offset dictionary for setence id
        offset_dict = dict()
        for l in open(y_text):
            tokens = l.split('|')
            id = tokens[0]
            offsets = [int(v) for v in tokens[1].split()]
            if id not in offset_dict.keys():
                offset_dict[id] = list()
            offset_dict[id].append(offsets)

        x_list = []
        y_list = []

        x_text_lines = [l.strip() for l in open(x_text).readlines()]
        max_sequence_length = max([len(x.split(' ')) for x in x_text_lines]) - 1 # -1:id token

        if not (os.path.exists(vocab_file)):
            vocab_dict = dict()
            vocab_size = 1  # the first vocab : 'NONE'
        else:
            vocab_dict = cPickle.load(open(vocab_file))
            vocab_size = len(vocab_dict.keys())

        for l in x_text_lines:
            tokens = l.split(' ')
            if tokens[0] in offset_dict:
                offsets = offset_dict[tokens[0]]
            else:
                offsets = []
            tokens = tokens[1:]
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

            #TODO: first word vector should be [0...0]
            sentence_vector = [0]*max_sequence_length

            # print 'max', max_sequence_length

            for i,t in enumerate(text_tokens):
                if t not in vocab_dict:
                    vocab_dict[t] = vocab_size
                    vocab_size += 1
                sentence_vector[i] = vocab_dict[t]

            x_list.append(sentence_vector)

            bio = [[0, 0, 1]]* len(tokens) + [[0, 0, 0]]*(max_sequence_length-len(tokens))
            for o in offsets:
                s = 0
                start_tag = None
                end_tag = None
                for i in range(len(tokens)):
                    if o[0] in range(s, s + len(tokens[i]) + 1):
                        start_tag = i
                    if o[1] in range(s, s + len(tokens[i]) + 1):
                        end_tag = i
                    s += len(tokens[i])

                for i in range(start_tag, end_tag + 1):
                    bio[i] = [0, 1, 0]
                bio[start_tag] = [1, 0, 0]
            y_list.append(bio)

        cPickle.dump(vocab_dict, open(vocab_file, 'w'))
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        return (np.array(x_list), np.array(y_list))

    def create_batches(self):
        self.num_batches = int(self.x.size / (self.batch_size *
                                                   self.max_sequence_length))

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.reshape(self.x, self.x.size)
        ydata = np.reshape(self.y, self.y.size)

        xdata = xdata[:self.num_batches * self.batch_size * self.max_sequence_length]
        ydata = ydata[:self.num_batches * self.batch_size * self.max_sequence_length * 3]

        print(self.num_batches, self.batch_size, self.max_sequence_length)

        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1, 3), self.num_batches, 1)


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
    pass