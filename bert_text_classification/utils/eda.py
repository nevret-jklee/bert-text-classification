import random
import os
import pandas as pd
import numpy as np

import random
import pyarrow.parquet as pq


''' Easy Data Augmentation'''
def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]

    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)

    return new_words

def text_aug(sentence, alpha_rs = 0.1, num_aug=3):
    words = sentence.split(' ')
    words = [word for word in words if word != ""]
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = num_aug

    n_rs = max(1, int(alpha_rs*num_words))

    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [sentence for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    return augmented_sentences


def eda():
    # bot = [26, 21, 31, 25, 9, 12, 10, 32, 11, 28, 27, 24, 29, 22, 30]
    bot = [22, 27, 28, 29, 30, 32]

    train = pq.read_table('').to_pandas()
    bot_data = train[train['label'].isin(bot)]
    
    aug = bot_data['target'].apply(lambda x: text_aug(x))

    tmp1 = bot_data.copy()
    tmp1['target'] = list(map(lambda x: x[0], aug))

    tmp2 = bot_data.copy()
    tmp2['target'] = list(map(lambda x: x[1], aug))

    tmp3 = bot_data.copy()
    tmp3['target'] = list(map(lambda x: x[2], aug))



if __name__ == '__main__':
    eda()
