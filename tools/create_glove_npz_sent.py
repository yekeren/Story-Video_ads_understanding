
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import logging

flags = tf.app.flags

flags.DEFINE_string('vocab_path', 'data/sentiments.txt', 
                    'Path to the vocabulary file.')

flags.DEFINE_string('output_npz_path', 'data/sent_vocab.npz', 
                    'Path to the output npz file.')

flags.DEFINE_string('data_path', 'zoo/glove.6B.200d.txt', 
                    'Path to the GLOVE data file.')

flags.DEFINE_float('init_width', 0.03, 
                   'Initial random width for unknown word.')

flags.DEFINE_boolean('unk_token', True, 
                     'If true, add UNK token.')

FLAGS = flags.FLAGS


def _load_vocab(filename):
  """Loads vocabulary from file.

  Args:
    filename: path to the vocabulary file.

  Returns:
    vocab: a list of words
  """
  vocab = []
  with open(filename, 'r') as fp:
    for line in fp.readlines():
      w = line.strip('\n')
      vocab.append(w)

  logging.info('Load %i words from %s.', len(vocab), filename)
  return vocab


def _load_data(filename):
  ''' Load a pre-trained word2vec file.

  Args:
    filename: path to the word2vec file.

  Returns:
    a numpy matrix.
  '''
  with open(filename, 'r') as fp:
    lines = fp.readlines()

  # Get the number of words and embedding size.
  num_words = len(lines)
  embedding_size = len(lines[0].strip('\n').split()) - 1

  word2vec = {}
  for line_index, line in enumerate(lines):
    items = line.strip('\n').split()
    word, vec = items[0], map(float, items[1:])
    assert len(vec) == embedding_size

    word2vec[word] = np.array(vec)
    if line_index % 10000== 0:
      logging.info('On load %s/%s', line_index, len(lines))
  return word2vec


def _export_data(word2vec, vocab, filename):
  """Export word embeddings to npz file.

  Args:
    word2vec: a mapping from word to vector.
    vocab: a mapping from word to id.
    filename: the name of output npz file.
  """
  words = vocab
  dims = word2vec['the'].shape[0]

  count = 0
  vecs = []
  for word in words:
    avg_vecs = []
    word_list = word.split(' ')
    for word in word_list:
      vec = word2vec[word]
      avg_vecs.append(vec)
    vec = np.stack(avg_vecs).mean(0)
    vecs.append(vec)
  vecs = np.stack(vecs)

  with open(filename, 'wb') as fp:
    np.save(fp, vecs)
  logging.info('Shape of word2vec: %s.', vecs.shape)
  logging.info('Unknown words: %i/%i(%i%%).', 
      count, len(vocab), int(count * 100 / len(vocab)))


def main(_):
  logging.set_verbosity(logging.INFO)

  vocab = _load_vocab(FLAGS.vocab_path)
  word2vec = _load_data(FLAGS.data_path)

  _export_data(word2vec, vocab, FLAGS.output_npz_path)

  logging.info('Done')

if __name__ == '__main__':
  app.run()
