
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def train_test_split(data_dict, test_rate=0.2):
    classes = list(data_dict.keys())

    train_set = defaultdict(list)
    test_set = defaultdict(list)
    for cls in classes:
        values = np.array(data_dict[cls])

        n_test = int(len(values)*test_rate)
        n_train = int(len(values)-n_test)

        train_indices = np.random.choice(len(values),n_train,replace=False)
        test_indices = np.setdiff1d(range(len(values)), train_indices)

        train_set[int(cls)] = values[train_indices, :]
        test_set[int(cls)] = values[test_indices, :]

    return train_set, test_set


def get_train_batch(data_dict, batch_size, n_batch, ref_dict=None, ref_key=None, zero_shot_cls=None):
    classes = list(data_dict.keys())

    if zero_shot_cls:
        classes.pop(int(zero_shot_cls))

    train_batch = []
    train_targets = []
    for n in range(n_batch):
        cls = np.random.randint(low=min(classes), high=max(classes), size=batch_size//2).tolist()
        pos_cls = cls
        neg_cls = [np.random.choice(np.setdiff1d(classes,cls[i]), size=1).item() for i in range(batch_size//2)]

        targets = np.ones((batch_size,1))
        targets[batch_size // 2:] = 0

        values = []
        pos_values = []
        neg_values = []
        for i in range(batch_size//2):

            value_idx = np.random.choice(len(data_dict[cls[i]]), size=1).item()
            pos_idx = np.random.choice(len(data_dict[pos_cls[i]]), size=1).item()
            neg_idx = np.random.choice(len(data_dict[neg_cls[i]]), size=1).item()
            
            value = data_dict[cls[i]][value_idx]
            if ref_dict:
                # with reference data
                if n%2==0:
                    pos_value = np.array(ref_dict[ref_key][str(pos_cls[i])]).squeeze(axis=0)
                    neg_value = np.array(ref_dict[ref_key][str(neg_cls[i])]).squeeze(axis=0)
                else:
                    pos_value = data_dict[pos_cls[i]][pos_idx]
                    neg_value = data_dict[neg_cls[i]][neg_idx]
            else:
                # only real data
                pos_value = data_dict[pos_cls[i]][pos_idx]
                neg_value = data_dict[neg_cls[i]][neg_idx]

            values.append(value)
            pos_values.append(pos_value)
            neg_values.append(neg_value)

        anchors = np.concatenate([values,values])
        comparison = np.concatenate([pos_values,neg_values])
        pairs = list(np.array([anchors, comparison]))

        train_batch.append(pairs)             # (n_batch, pairs, batch_size, length, point)
        train_targets.append(targets)         # (n_batch, target)

    return train_batch, train_targets


def get_test_batch(data_dict, ref_dict, ref_key, zero_shot_cls=None):
    classes = list(data_dict.keys())
    n_class = len(classes)

    if zero_shot_cls:
        classes.pop(int(zero_shot_cls))

    test_batch = []
    test_targets = []
    for cls in classes:
        for value in data_dict[cls]:
            anchors = np.array([value]*n_class)
            comparison = np.array(list(ref_dict[ref_key].values())).squeeze(axis=1)
            
            pairs = list(np.array([anchors, comparison]))
            targets = np.zeros((n_class,1))
            if ref_key == 'katakana':
                targets[cls-1] = 1
            else:
                targets[cls] = 1

            test_batch.append(pairs)
            test_targets.append(targets)

    return test_batch, test_targets

def get_zero_shot_batch(data_dict, ref_dict, ref_key, zero_shot_cls):
    classes = list(data_dict.keys())
    n_class = len(classes)

    zero_shot_batch = []
    zero_shot_targets = []
    for value in data_dict[zero_shot_cls]:
        anchors = np.array([value]*n_class)
        comparison = np.array(list(ref_dict[ref_key].values())).squeeze(axis=1)
        
        pairs = list(np.array([anchors, comparison]))
        targets = np.zeros((n_class,1))
        targets[int(zero_shot_cls)]=1

        zero_shot_batch.append(pairs)
        zero_shot_targets.append(targets)

    return zero_shot_batch, zero_shot_targets


def plot_images(input_list, task='train'):
  row, col, _, _ = np.array(input_list).shape
  fig,axes = plt.subplots(row,col, figsize=(col//row*4,row*2))
  n = 0
  for r in range(row):
    for c in range(col):
      x,y = zip(*input_list[r][c])

      axes[r][c].set_xlim([-0.2, 1.2])
      axes[r][c].set_ylim([-0.2, 1.2])
      axes[r][c].get_xaxis().set_visible(False)
      axes[r][c].get_yaxis().set_visible(False)

      if task=='test':
        axes[r][c].scatter(x,y, 1)
      else:
        if c >= col//2:
          axes[r][c].scatter(x,y, 1, color='red')
        else:
          axes[r][c].scatter(x,y, 1, color='blue')
  n += 1