import os
import sys
import re
import json
import string
import random

import numpy as np

printable = set(string.printable)
convert_to_printable = lambda caption: filter(lambda x: x in printable, caption)

class AdsApi(object):
  def __init__(self, conf, invalid_items=[]):
    """Initialize api from config file.

    Args:
      config: a config file in json format.
    """
    with open(conf, 'r') as fp:
      config = json.loads(fp.read())

    train_ids, valid_ids, test_ids = self._get_splits(
        config['train_ids'], config['valid_ids'], config['test_ids'])

    # Initialize meta info.
    assert len(set(train_ids) & set(valid_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(valid_ids) & set(test_ids)) == 0

    self._meta = {}
    self._meta.update([(x, {'split': 'train'}) for x in train_ids])
    self._meta.update([(x, {'split': 'valid'}) for x in valid_ids])
    self._meta.update([(x, {'split': 'test'}) for x in test_ids])

    for image_id in self._meta:
      self._meta[image_id].update({
        'image_id': image_id,
        'file_path': os.path.join(config['image_path'], image_id),
      })
    print >> sys.stderr, 'Load %d examples.' % (len(self._meta))

    if 'topic_list' in config and 'topic_annotations' in config:
      self._process_topic(config['topic_list'], config['topic_annotations'])

    if 'symbol_list' in config and 'symbol_annotations' in config:
      self._process_symbol(config['symbol_list'], config['symbol_annotations'])

    if 'qa_annotations' in config:
      self._process_qa(config['qa_annotations'])

    if 'action_annotations' in config and 'reason_annotations' in config:
      self._process_seperate_action_reason(
          config['action_annotations'],
          config['reason_annotations'],
          config.get('num_positive_statements', None))

    if 'action_reason_annotations' in config:
      self._process_combined_action_reason(
          config['action_reason_annotations'],
          config.get('num_positive_statements', None))

    if 'num_negative_statements' in config:
      self._sample_negative_statements(
          config['num_negative_statements'])

    self._summerize()

  def get_meta_by_id(self, image_id):
    """Returns meta info based on image_id.

    Args:
      image_id: string image_id.

    Returns:
      meta: meta info.
    """
    return self._meta[image_id]

  def get_meta_list(self, split=None):
    """Get meta list.

    Args:
      split: could be one of 'train', 'valid', 'test'.

    Returns:
      meta_list: meta list for the specific split.
    """
    meta_list = self._meta.values()
    
    if split is not None:
      assert split in ['train', 'valid', 'test']
      meta_list = [x for x in meta_list if x['split'] == split]

    return meta_list

  def _summerize(self):
    """Print statistics to the standard error.
    """
    meta_list = self.get_meta_list()

    num_topic_annots = len([x for x in meta_list if 'topic_id' in x])
    print >> sys.stderr, '%d examples associate with topics.' % (num_topic_annots)

    num_symb_annots = len([x for x in meta_list if 'symbol_ids' in x])
    print >> sys.stderr, '%d examples associate with symbols.' % (num_symb_annots)

    num_stmt_annots = len([x for x in meta_list if 'actions_and_reasons' in x])
    print >> sys.stderr, '%d examples associate with statements.' % (num_stmt_annots)

    num_neg_stmt_annots = len([x for x in meta_list if 'negative_actions_and_reasons' in x])
    print >> sys.stderr, '%d examples associate with negative statements.' % (num_neg_stmt_annots)

  def get_answer_vocab(self):
    """Get ansewr vocabulary.

    Returns:
      vocab: a list of words.
    """
    return self._answer_vocab

  def _get_splits(self, train_ids_file, valid_ids_file, test_ids_file):
    """Get splits from pre-partitioned file.

    Args:
      train_ids_file: file containing train ids.
      valid_ids_file: file containing valid ids.
      test_ids_file: file containing test ids.
    """
    with open(train_ids_file, 'r') as fp:
      train_ids = [x.strip() for x in fp.readlines()]
    with open(valid_ids_file, 'r') as fp:
      valid_ids = [x.strip() for x in fp.readlines()]
    with open(test_ids_file, 'r') as fp:
      test_ids = [x.strip() for x in fp.readlines()]

    print >> sys.stderr, 'Load %d train examples.' % (len(train_ids))
    print >> sys.stderr, 'Load %d valid examples.' % (len(valid_ids))
    print >> sys.stderr, 'Load %d test examples.' % (len(test_ids))
    return train_ids, valid_ids, test_ids

  def _majority_vote(self, elems):
    """Process majority votes.

    Args:
      elems: a list of elems.

    Returns:
      elem: the element who gets the most votes.
      num_votes: number of votes.
    """
    votes = {}
    for e in elems:
      votes[e] = votes.get(e, 0) + 1
    votes = sorted(votes.iteritems(), lambda x, y: -cmp(x[1], y[1]))
    return votes[0]

  def _process_topic(self, topic_list, topic_annotations):
    """Process topic annotations.

    Modifies 'topic_id' data field in the meta data.
    Modifies 'topic_name' data field in the meta data.

    Args:
      topic_list: path to the file storing topic list.
      topic_annotations: path to the topic json file.
    """

    def _revise_topic_id(topic_id):
      """Revises topic id.
      Args:
        topic_id: topic id in string format.
      Returns:
        topic_id: topic id in number format.
      """
      if not topic_id.isdigit():
        return None
      topic_id = int(topic_id)
      if topic_id == 39: topic_id = 0
      return topic_id

    def _revise_topic_name(name):
      """Revises topic name.
      Args:
        topic_name: topic name in long description format.
      Returns:
        topic_name: short topic name.
      """
      matches = re.findall(r"\"(.*?)\"", name)
      if len(matches) > 1:
        return matches[1].lower()
      return matches[0].lower()

    self._topic_to_name = {}
    with open(topic_list, 'r') as fp:
      for line in fp.readlines():
        topic_id, topic_name = line.strip('\n').split('\t') 

        topic_id = _revise_topic_id(topic_id)
        topic_name = _revise_topic_name(topic_name)
        if topic_id is None:
          raise ValueError('Invalid topic id %s.' % (topic_id))
        self._topic_to_name[topic_id] = topic_name

    with open(topic_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    for image_id, topic_id_list in annots.iteritems():
      meta = self._meta[image_id]
      topic_id_list = [_revise_topic_id(tid) for tid in topic_id_list]
      topic_id_list = [tid for tid in topic_id_list if tid is not None]

      if len(topic_id_list) > 0:
        topic_id, num_votes = self._majority_vote(topic_id_list)

        meta['topic_id'], meta['topic_votes'] = topic_id, num_votes
        meta['topic_name'] = self._topic_to_name[topic_id]

  def _process_symbol(self, symbol_list, symbol_annotations):
    """Process symbol annotations.

    Modifies 'symbol_ids' data field in the meta data.
    Modifies 'symbol_names' data field in the meta data.

    Args:
      symbol_list: path to the file storing symbol list.
      symbol_annotations: path to the symbol json file.
    """
    with open(symbol_list, 'r') as fp:
      data = json.loads(fp.read())

    self._symbol_to_name = {0: 'unclear'}
    symbol_to_id = {}
    for cluster in data['data']:
      self._symbol_to_name[cluster['cluster_id']] = cluster['cluster_name']
      for symbol in cluster['symbols']:
        symbol_to_id[symbol] = cluster['cluster_id']
      print >> sys.stderr, cluster['cluster_name']

    with open(symbol_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    for image_id, objects in annots.iteritems():
      meta = self._meta[image_id]
      symbol_list = []
      for obj in objects:
        symbols = [s.strip() for s in obj[4].lower().split('/') if len(s.strip()) > 0]
        symbols = [symbol_to_id[s] for s in symbols if s in symbol_to_id]
        symbol_list.extend(symbols)

      if len(symbol_list) > 0:
        symbol_list = list(set(symbol_list))
        meta['symbol_ids'] = symbol_list
        meta['symbol_names'] = [self._symbol_to_name[s] for s in symbol_list]

  def _process_qa(self, qa_annotations):
    """Process qa annotations.

    Modifies 'single_word_answers' data field in the meta data.

    Args:
      qa_annotations: path to the qa annotations json file.
    """
    with open(qa_annotations, 'r') as fp:
      data = json.loads(fp.read())
    
    self._answer_vocab = data['vocab'] 
    for image_id, qa_annotation in data['annotations'].iteritems():
      meta = self._meta[image_id]
      meta['qa_question'] = qa_annotation['question']
      meta['single_word_answers'] = qa_annotation['answer']

  def _revise_statement_to_action_and_reason(self, statement):
    """Revises statement to action and reason.

    Args:
      statement: a string similar to "I should buy a car because it is stable
      and strong".

    Returns:
      action: a string similar to "I should buy a car".
      reason: a string similar to "Because it is cheap."
    """
    pos = statement.lower().find('because')
    if pos < 0:
      return None, None

    action = statement[:pos].strip().strip('.').lower()
    reason = statement[pos:].strip().strip('.').lower()

    if action[:len("i should")] == "i should" \
      or reason[:len("because")] == "because":
      return action, reason
    return None, None

  def _process_combined_action_reason(self,
      qa_action_reason_annotations, num_positive_statements):
    """Processes action reason annotations.

    Modifies 'statements' data field in the meta data.

    Args:
      qa_action_reason_annotations: a file containing qa action-reason
      annotations.
      num_positive_statements: number of positive statements needed.

    Raises:
      ValueError: if annotations file is invalid.
    """
    # Read file.
    with open(qa_action_reason_annotations, 'r') as fp:
      annots = json.loads(fp.read())

    # Parse annotations.
    for image_id, statements in annots.iteritems():
      meta = self._meta[image_id]

      # Process statements.
      statements = [convert_to_printable(statement) for statement in statements]

      # Process actions and reasons.
      actions_and_reasons = []
      for statement in statements:
        action, reason = self._revise_statement_to_action_and_reason(statement)
        if action and reason:
          actions_and_reasons.append((action, reason))

      if len(actions_and_reasons) > 0:
        if num_positive_statements is None:
          meta['actions_and_reasons'] = actions_and_reasons
        elif len(actions_and_reasons) >= num_positive_statements:
          meta['actions_and_reasons'] = actions_and_reasons[:num_positive_statements]


  def _process_seperate_action_reason(self, 
      qa_action_annotations, qa_reason_annotations,
      num_positive_statements):
    """Process action reason annotations.

    Modifies 'actions' and 'reasons' data field in the meta data.

    Args:
      qa_action_annotations: a file containing qa action annotations.
      qa_reason_annotations: a file containing qa reason annotations.
      num_positive_statements: number of positive statements needed.
    """
    # Read files.
    with open(qa_action_annotations, 'r') as fp:
      actions = sorted(
          json.loads(fp.read()).iteritems(), 
          lambda x, y: cmp(x[0], y[0]))

    with open(qa_reason_annotations, 'r') as fp:
      reasons = sorted(
          json.loads(fp.read()).iteritems(), 
          lambda x, y: cmp(x[0], y[0]))

    assert len(actions) == len(reasons)

    # Process qa annotations.
    for (action_id, action_annots), (reason_id, reason_annots) in zip(actions, reasons):
      assert action_id == reason_id
      assert len(action_annots) == len(reason_annots)

      meta = self._meta[action_id]
      actions_and_reasons = []

      for action, reason in zip(action_annots, reason_annots):
        action = convert_to_printable(action).strip().strip('.').lower()
        reason = convert_to_printable(reason).strip().strip('.').lower()

        if action and reason:
          if action[:len("i should")] == "i should" \
            or reason[:len("because")] == "because":
            actions_and_reasons.append((action, reason))

      if len(actions_and_reasons) > 0:
        if num_positive_statements is None:
          meta['actions_and_reasons'] = actions_and_reasons
        elif len(actions_and_reasons) >= num_positive_statements:
          meta['actions_and_reasons'] = actions_and_reasons[:num_positive_statements]

  def _sample_negative_statements(self, negative_examples_per_image):
    """Randomly sample negative statements, only for evaluation purpose.

    Modify 'negative_statements' data field.

    Args:
      negative_examples_per_image: number of negative examples of each image.
    """
    random.seed(286)
    meta_list = [meta for meta in self.get_meta_list() \
                if 'actions_and_reasons' in meta]

    for i, meta in enumerate(meta_list):
      neg_stmts = []
      for _ in xrange(negative_examples_per_image):
        index = random.randint(1, len(meta_list) - 1)
        index = (i + index) % len(meta_list)
        assert index != i
        stmts = meta_list[index]['actions_and_reasons']
        index = random.randint(0, len(stmts) - 1)
        neg_stmts.append(stmts[index])

      meta['negative_actions_and_reasons'] = neg_stmts

#if __name__ == '__main__':
#  api = AdsApi('configs/ads_api.config.0')
