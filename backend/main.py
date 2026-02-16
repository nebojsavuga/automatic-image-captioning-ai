import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

from preprocess import preprocess_text, preprocess_image, get_captions_with_file_names
from image_feature_extraction import feature_extractor
from model import get_model

# Load captions
captions_df = get_captions_with_file_names("images/results.csv")

# Preprocess captions
captions_df['caption'] = captions_df['caption'].apply(preprocess_text)


# Load preprocessed images directly (option)
# image_features = np.load("images/image_features.npy")
image_features = np.load("images/preprocessed_images.npy")

# Prepare vocabulary
all_captions = captions_df['caption'].values
words = set()
for c in all_captions:
	words.update(c.split())
word2idx = {w: i+1 for i, w in enumerate(sorted(words))}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx) + 1

# Prepare sequences
max_length = max(len(c.split()) for c in all_captions)
sequences = []
for c in all_captions:
	seq = [word2idx[w] for w in c.split()]
	sequences.append(seq)
sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split dataset
split1 = int(0.7 * len(sequences))
split2 = int(0.9 * len(sequences))
X_train, X_val, X_test = image_features[:split1], image_features[split1:split2], image_features[split2:]
y_train, y_val, y_test = sequences[:split1], sequences[split1:split2], sequences[split2:]

# Build model
model = get_model(max_length, vocab_size)

# Prepare targets for training
def create_targets(seqs, vocab_size):
	X = []
	y = []
	for seq in seqs:
		for i in range(1, len(seq)):
			X.append(seq[:i])
			y.append(seq[i])
	X = pad_sequences(X, maxlen=max_length, padding='post')
	y = to_categorical(y, num_classes=vocab_size)
	return X, y

X_seq_train, y_seq_train = create_targets(y_train, vocab_size)

# Train model (simplified, for demo)
model.fit([X_train, X_seq_train], y_seq_train, epochs=2, batch_size=32)

# Generate captions (greedy)
def generate_caption(feature, model, word2idx, idx2word, max_length):
	in_text = 'startseq'
	for _ in range(max_length):
		sequence = [word2idx.get(w, 0) for w in in_text.split()]
		sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
		yhat = model.predict([feature.reshape(1, -1), sequence], verbose=0)
		yhat_idx = np.argmax(yhat)
		word = idx2word.get(yhat_idx, '')
		if word == 'endseq' or word == '':
			break
		in_text += ' ' + word
	return in_text.replace('startseq', '').replace('endseq', '').strip()

# Evaluate BLEU, ROUGE, METEOR
references = [[c.split()] for c in captions_df['caption'].values[split2:]]
generated = []
for feat in X_test:
	cap = generate_caption(feat, model, word2idx, idx2word, max_length)
	generated.append(cap.split())
bleu_score = corpus_bleu(references, generated)
rouge = Rouge()
rouge_score = rouge.get_scores([' '.join(g) for g in generated], [' '.join(r[0]) for r in references], avg=True)
meteor_scores = [meteor_score([r[0]], ' '.join(g)) for r, g in zip(references, generated)]
avg_meteor = np.mean(meteor_scores)

print(f"BLEU: {bleu_score:.4f}")
print(f"ROUGE: {rouge_score}")
print(f"METEOR: {avg_meteor:.4f}")

# Visualize results
for i in range(3):
	img_idx = split2 + i
	img_name = captions_df.iloc[img_idx]['image']
	img_path = f"images/flickr30k_images/{img_name}"
	img = plt.imread(img_path)
	plt.imshow(img)
	plt.title(' '.join(generated[i]))
	plt.axis('off')
	plt.show()