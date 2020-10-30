
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import logging

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('mscoco_vocab_path', 'configs/mscoco_vocab.txt', 
                    'Path to the vocabulary file.')

flags.DEFINE_string('place_vocab_path', 'configs/categories_places365.txt', 
                    'Path to the vocabulary file.')

flags.DEFINE_string('sentiment_vocab_path', 'data/sentiments.txt', 
                    'Path to the vocabulary file.')

flags.DEFINE_string('data_path', 'zoo/glove.6B.200d.txt', 
                    'Path to the GLOVE data file.')

flags.DEFINE_string('coco_sent_csv_path', 'tmp/coco_sent.csv', 
                    'Path to the output confusion matrix file.')

flags.DEFINE_string('place_sent_csv_path', 'tmp/place_sent.csv', 
                    'Path to the output confusion matrix file.')

flags.DEFINE_string('emotic_sent_csv_path', 'tmp/emotic_sent.csv', 
                    'Path to the output confusion matrix file.')

age_vocab = [ "Adult", "Teenager", "Kid" ]
gender_vocab = [ "Male", "Female" ]
category_vocab = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence",
  "Happiness", "Pleasure", "Excitement", "Surprise", "Sympathy",
  "Confusion", "Disconnection", "Fatigue", "Embarrassment", "Yearning",
  "Disapproval", "Aversion", "Annoyance", "Anger", "Sensitivity", "Sadness",
  "Disquiet", "Fear", "Pain", "Suffering"]

def _load_mscoco_vocab(filename):
  """Loads vocabulary from file.

  Args:
    filename: path to the vocabulary file.

  Returns:
    vocab: a list of (id, word) tuples.
  """
  vocab = []
  with open(filename, 'r') as fp:
    for line in fp.readlines():
      wid, w = line.strip('\n').split('\t')
      vocab.append((int(wid), w))

  logging.info('Load %i words from %s.', len(vocab), filename)
  return vocab

def _load_place_vocab(filename):
  """Loads vocabulary from file.

  Args:
    filename: path to the vocabulary file.

  Returns:
    vocab: a list of (id, word) tuples.
  """
  vocab = []
  with open(filename, 'r') as fp:
    for _, line in enumerate(fp.readlines()):
      w, wid = line.strip('\n').split(' ')
      w, wid = w[3:], int(wid)
      w = w.replace('/', ' ')
      w = w.replace('_', ' ')
      w = w.replace('barndoor', 'barn door')
      w = w.replace('kindergarden', 'kindergarten')
      w = w.replace('oilrig', 'oil rig')
      vocab.append((wid, w))

  logging.info('Load %i words from %s.', len(vocab), filename)
  return vocab

def _load_vocab(filename):
  """Loads vocabulary from file.

  Args:
    filename: path to the vocabulary file.

  Returns:
    vocab: a list of (id, word) tuples.
  """
  vocab = []
  with open(filename, 'r') as fp:
    for wid, line in enumerate(fp.readlines()):
      w = line.strip('\n')
      vocab.append((wid, w))

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

def _load_embedding(word2vec, vocab):
  """Load word embedding.

  Args:
    word2vec: a mapping from word to vector.
    vocab: a mapping from word to id.

  Returns:
    w_id_list: a list of word ids.
    w_list: a list of words.
    emb_list: a numpy array.
  """
  w_id_list, w_list, emb_list = [], [], []
  for w_id, word in vocab:
    avg_vecs = []
    word_list = word.split(' ')
    for word in word_list:
      vec = word2vec[word]
      avg_vecs.append(vec)
    emb_list.append(np.stack(avg_vecs).mean(0))
    w_id_list.append(w_id)
    w_list.append('_'.join(word_list))
  return w_id_list, w_list, np.stack(emb_list, axis=0)


def _l2_distance(x, y):
  return np.sqrt(np.sum((x-y)**2))

def _cosine_similarity(x, y):
  x = x / np.linalg.norm(x) 
  y = y / np.linalg.norm(y)
  return (x * y).sum()


def _compute_confusion_matrix(
    words1, embs1, words2, embs2, filename, distance_fn=_cosine_similarity):

  mat = np.zeros((len(words1), len(words2)), np.float32)
  with open(filename, 'w') as fp:
    for word2, emb2 in zip(words2, embs2):
      fp.write(',' + word2)
    fp.write('\n')
    for ind1, (word1, emb1) in enumerate(zip(words1, embs1)):
      fp.write(word1)
      for ind2, (word2, emb2) in enumerate(zip(words2, embs2)):
        dist = distance_fn(emb1, emb2)
        mat[ind1, ind2] = dist
        fp.write(',%.4lf' % (dist))
      fp.write('\n')
  logging.info('Confusion matrix is exported to %s.', filename)
  return mat


def _save_coco(word_ids, words, embs, filename):
  mat = np.zeros((90, embs.shape[1]), np.float32)
  for word_id, emb in zip(word_ids, embs):
    mat[word_id] = emb

  with open(filename, 'wb') as fp:
    np.save(fp, mat)

  logging.info('Matrix is saved to %s, shape=%s.', filename, mat.shape)

def _save_mat(embs, filename):
  with open(filename, 'wb') as fp:
    np.save(fp, embs)

  logging.info('Matrix is saved to %s, shape=%s.', filename, embs.shape)



def main(_):
  logging.set_verbosity(logging.INFO)

  # Load vocab.
  mscoco_vocab = _load_mscoco_vocab(FLAGS.mscoco_vocab_path)
  place_vocab = _load_place_vocab(FLAGS.place_vocab_path)
  sentiment_vocab = _load_vocab(FLAGS.sentiment_vocab_path)
  emotic_vocab = age_vocab + gender_vocab + category_vocab
  emotic_vocab = [(i, word.lower()) for i, word in enumerate(emotic_vocab)]

  # Assign embedding vectors.
  word2vec = _load_data(FLAGS.data_path)
  coco_w_id, coco_w, coco_embs = _load_embedding(word2vec, mscoco_vocab)
  plac_w_id, plac_w, plac_embs = _load_embedding(word2vec, place_vocab)
  sent_w_id, sent_w, sent_embs = _load_embedding(word2vec, sentiment_vocab)
  emotic_w_id, emotic_w, emotic_embs = _load_embedding(word2vec, emotic_vocab)

  sent_coco_mat = _compute_confusion_matrix(
      sent_w, sent_embs, coco_w, coco_embs, FLAGS.coco_sent_csv_path)
  sent_plac_mat = _compute_confusion_matrix(
      sent_w, sent_embs, plac_w, plac_embs, FLAGS.place_sent_csv_path)
  sent_emotic_mat = _compute_confusion_matrix(
      sent_w, sent_embs, emotic_w, emotic_embs, FLAGS.emotic_sent_csv_path)

  # Save embeddings.
  _save_coco(coco_w_id, coco_w, coco_embs, 'tmp/coco.emb.npz.200d')
  _save_coco(coco_w_id, coco_w, sent_coco_mat.transpose(), 'tmp/coco_sent.emb.npz.200d')
  _save_coco(coco_w_id, coco_w, 
      np.concatenate([coco_embs, sent_coco_mat.transpose()], axis=1), 'tmp/coco_comb.emb.npz.200d')

  _save_mat(plac_embs, 'tmp/place.emb.npz.200d')
  _save_mat(sent_plac_mat.transpose(), 'tmp/place_sent.emb.npz.200d')
  _save_mat(np.concatenate([plac_embs, sent_plac_mat.transpose()], axis=1), 
      'tmp/place_comb.emb.npz.200d')

  _save_mat(emotic_embs, 'tmp/emotic.emb.npz.200d')
  _save_mat(sent_emotic_mat.transpose(), 'tmp/emotic_sent.emb.npz.200d')
  _save_mat(np.concatenate([emotic_embs, sent_emotic_mat.transpose()], axis=1), 
      'tmp/emotic_comb.emb.npz.200d')

  _save_mat(sent_embs, 'tmp/sent.emb.npz.200d')

  logging.info('Done')

if __name__ == '__main__':
  app.run()
