from typing import Dict, List


def build_vocabulary(
    caption_mapping: Dict[str, List[str]], train_images: List[str]
):
    words = set()
    for image_name in train_images:
        for caption in caption_mapping[image_name]:
            words.update(caption.split())

    word2idx = {word: idx + 1 for idx, word in enumerate(sorted(words))}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocabulary_size = len(word2idx) + 1
    return word2idx, idx2word, vocabulary_size


def get_max_length(caption_mapping: Dict[str, List[str]], train_images: List[str]):
    max_length = 0
    for image_name in train_images:
        for caption in caption_mapping[image_name]:
            max_length = max(max_length, len(caption.split()))
    return max_length
