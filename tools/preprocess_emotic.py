import os
import sys
import json
import argparse


def _preprocess(filename):
  result = {}
  with open(filename, 'r') as fp:
    data = json.loads(fp.read())

  result['filename'] = data['filename']
  result['folder'] = data['folder']

  result['height'] = data['image_size']['n_row']
  result['width'] = data['image_size']['n_col']

  # Person list.
  if type(data['person']) == dict: 
    person_list = [data['person']]
  elif type(data['person']) == list: 
    person_list = data['person']

  result['person'] = []

  for person in person_list:
    # We care only about age, gender, categories, and bounding box.
    categories = []
    age = person['age']
    gender = person['gender']
    body_bbox = person['body_bbox']

    if type(person['annotations_categories']) == dict:
      annotations_categories_list = [person['annotations_categories']]
    elif type(person['annotations_categories']) == list:
      annotations_categories_list = person['annotations_categories']

    for annotations_category in annotations_categories_list:
      categories.extend(annotations_category['categories'])

    categories = list(set(categories))
    result['person'].append({
        'body_bbox': body_bbox,
        'age': age,
        'gender': gender,
        'categories': categories,
        })
  return result


def main(argv):
  anno_list = []
  for filename in os.listdir(argv['data_dir']):
    if filename[-4:] == '.txt':
      filename = os.path.join(argv['data_dir'], filename)
      anno = _preprocess(filename)
      anno_list.append(anno)

  with open(argv['json_output'], 'w') as fp:
    fp.write(json.dumps(anno_list))

  print >> sys.stderr, 'Dumps %i records to %s.' % (
      len(anno_list), argv['json_output'])
  print >> sys.stderr, 'Done'

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', default='EMOTIC/train', help='')
  parser.add_argument('--json_output', default='EMOTIC/train.json', help='')

  argv = vars(parser.parse_args())
  print >> sys.stderr, 'parsed input parameters:'
  print >> sys.stderr, json.dumps(argv, indent = 2)

  main(argv)

  print >> sys.stderr, 'Done'
