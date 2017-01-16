import random

f = open("/mnt/smb/Q/sunkyu/biomedical_ner_res/f_1900.txt.biotag")

unique_labels=set()
data = []

data_size = 100000
for l in f:
    text, label = l.split("\t")

    ll = label.strip().split(",")
    if len(set(ll)) == 1:
        continue

    data.append([text.strip(), label.strip()])
    for v in set(ll):
        unique_labels.add(v)

    if len(data) == data_size:
        break

random.shuffle(data)
train_index = int(data_size*0.8)

with open("res/Pubmed/data.txt", "w") as fw:
    for d in data:
        fw.write(d[0] + "\t" + d[1] + "\n")

with open("res/Pubmed/train/text.txt", "w") as fw:
    for d in data[:train_index]:
        fw.write(d[0] + "\t" + d[1] + "\n")

with open("res/Pubmed/test/text.txt", "w") as fw:
    for d in data[train_index:]:
        fw.write(d[0] + "\t" + d[1] + "\n")