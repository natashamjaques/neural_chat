"""Script for downloading Universal Sentence Encoder used for calculating
Semantic Similarity Reward"""
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub


# Set up download dir
project_dir = Path(__file__).resolve().parent.parent
print(project_dir)
exit()
USE_dir = project_dir.joinpath('UniversalSentenceEncoder')
os.environ["TFHUB_CACHE_DIR"] = USE_dir

# Download encoder
encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

# Check for errors
embeddings = encoder([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embs = sess.run(embeddings)
    assert(embs.shape == (2, 512))


print('#' * 10)
print('#' * 10)
print('Done installing Universal Sentence Encoder')
print('#' * 10)
print('#' * 10)
