import numpy as np


def levenshtein(s1, s2):
    """
    Calculate Levenshtein distance between two strings.
    """
    distance = [[0 for _ in range(len(s2) + 1)]
                for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i == 0:
                distance[i][j] = j
            elif j == 0:
                distance[i][j] = i
            else:
                diag = distance[i - 1][j - 1] + (s1[i - 1] != s2[j - 1])
                upper = distance[i - 1][j] + 1
                left = distance[i][j - 1] + 1
                distance[i][j] = min(diag, upper, left)

    return distance[len(s1)][len(s2)]


def wer(labels, pred_labels):
    """
    Calculate Word Error Rate (WER).
    WER compares hypothesis to reference text.
    WER is defined as: (Sw + Dw + Iw) / N
    Sw is the number of words substituted,
    Dw is the number of words deleted,
    Iw is the number of words inserted,
    and N is the number of words in the reference.
    """
    assert len(labels) == len(pred_labels)
    lev_distances, num_words = 0, 0
    for pred_label, label in zip(pred_labels, labels):
        words, pred_words = label.split(), pred_label.split()
        lev_distances += levenshtein(pred_words, words)
        num_words += len(words)
    return lev_distances / num_words


def cer(labels, pred_labels):
    """
    Calculate Character Error Rate (CER).
    CER compares hypothesis to reference text.
    """
    assert len(labels) == len(pred_labels)
    lev_distances, num_chars = 0, 0
    for pred_label, label in zip(pred_labels, labels):
        lev_distances += levenshtein(pred_label, label)
        num_chars += len(label)
    return lev_distances / num_chars


def accuracy(labels, pred_labels):
    """
    Calculate accuracy of the model.
    """
    assert len(labels) == len(pred_labels)
    return np.sum(np.compare_chararrays(labels, pred_labels, "==", False)) / len(labels)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
