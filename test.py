
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
import pickle

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='res/BioCreative2GM/test/', help='data directory')
    parser.add_argument('--data_dir', type=str, default='res/Pubmed/test/', help='data directory')
    parser.add_argument('--restore', type=str, default='None', help='ckpt file path')
    parser.add_argument('--save_dir', type=str, default='logs/pubmed/', help='ckpt file path')
    parser.add_argument('--batch_size', type=int, default=128, help='data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='num_epoch')
    parser.add_argument('--rnn_size', type=int, default=100, help='output nodes of rnn')
    parser.add_argument('--class_size', type=int, default=17, help='class size')

    args = parser.parse_args()
    procname.setprocname("NER_Pubmed_TEST")
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
    get_f1_score(prediction, data_loader.x_text, data_loader.y_text, data_loader.label_dict)
    # tag(prediction, data_loader.x_text, data_loader.y_text, data_loader.label_dict)

def get_f1_score(prediction, x_text, y_text, label_dict):
    pred_int_labels = []
    for p_s in prediction:
        pred_int_labels.append([list(p).index(max(p)) for p in p_s])

    result_list = []
    for i in range(len(y_text)):
        x_one_sentence = x_text[i]
        y_label_one_sentence = y_text[i]
        pred_int_label_one_sentence = pred_int_labels[i]
        pred_label_one_sentence = []
        for j in range(len(x_one_sentence)):
            pred_label = 'NONE'
            for k in label_dict:
                if np.argmax(np.array(label_dict[k])) == pred_int_label_one_sentence[j]:
                    pred_label = k
            pred_label_one_sentence.append(pred_label)

        y_entities = []
        temp_text_list = []
        temp_type_list = []
        token_index = 0
        while True:
            if token_index >= len(x_one_sentence):
                break
        # for token_index in range(len(y_label_one_sentence)):
            y_label_tokens = y_label_one_sentence[token_index].split("_")
            if len(y_label_tokens) > 1: #type_B or type_I
                if y_label_tokens[1] == 'B': #type_B
                    while True:
                        temp_text_list.append(x_one_sentence[token_index])
                        temp_type_list.append(y_label_tokens[0])
                        token_index += 1
                        if token_index == len(y_label_one_sentence):
                            break
                        y_label_tokens = y_label_one_sentence[token_index].split("_")

                        if len(y_label_tokens) == 1: #'O'
                            if len(set(temp_type_list)) == 1:
                                y_entities.append((' '.join(temp_text_list),temp_type_list[0]))
                            temp_text_list = []
                            temp_type_list = []
                            break
            token_index += 1

        pred_entities = []
        temp_text_list = []
        temp_type_list = []

        token_index = 0
        while True:
            if token_index >= len(x_one_sentence):
                break
        # for token_index in range(len(pred_label_one_sentence)):
            pred_label_tokens = pred_label_one_sentence[token_index].split("_")
            if len(pred_label_tokens) > 1: #type_B or type_I
                if pred_label_tokens[1] == 'B': #type_B
                    while True:
                        temp_text_list.append(x_one_sentence[token_index])
                        temp_type_list.append(pred_label_tokens[0])
                        token_index += 1
                        if token_index == len(pred_label_one_sentence):
                            break
                        pred_label_tokens = pred_label_one_sentence[token_index].split("_")

                        if len(pred_label_tokens) == 1: #'O'
                            if len(set(temp_type_list)) == 1:
                                pred_entities.append((' '.join(temp_text_list),temp_type_list[0]))
                            temp_text_list = []
                            temp_type_list = []
                            break
            token_index += 1
        result_list.append({'Dictionary':y_entities, 'Prediction':pred_entities})
    pickle.dump(result_list, open("temp_result.pickle","w"))
    __get_f1_score()

def __get_f1_score():
    result_list = pickle.load(open("temp_result.pickle"))

    num_pred = 0
    num_entity = 0
    not_found_list = []
    more_found_list = []
    for d in result_list:
        entity = d['Dictionary']
        pred = d['Prediction']
        num_entity += len(entity)
        num_pred += len(pred)

        not_found = [e for e in entity if e not in pred]
        more_found = [e for e in pred if e not in entity]
        # print(not_found, more_found)
        not_found_list += not_found
        more_found_list += more_found

    print(num_entity, num_pred, len(not_found_list), len(more_found_list))

    for v in not_found_list[:30]:
        print(v)
    for v in more_found_list[:30]:
        print(v)

def tag(prediction, x_text, y_text, label_dict):
    o = []
    for p_s in prediction:
        o.append([list(p).index(max(p)) for p in p_s])

    fw = open("result.txt", "w")
    for i in range(len(o)):
        x_l = x_text[i]
        y_l = y_text[i]
        o_l = o[i]
        for j in range(len(x_l)):
            if o_l[j] != 10:
                o_l_text = 'NONE'
                for k in label_dict:
                    if np.argmax(np.array(label_dict[k])) == o_l[j]:
                        o_l_text = k
                fw.write(",".join([x_l[j], y_l[j], o_l_text]))
                fw.write("\n")
    fw.close()
if __name__ == '__main__':
    # main()
    __get_f1_score()