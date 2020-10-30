import os
import sys
import json
import random
import argparse

random.seed(286)

def main(argv):
  with open(argv['video_list'], 'r') as fp:
    video_ids = [line.strip('\n\'') for line in fp.readlines()]

  random.shuffle(video_ids)

  num_splits = argv['num_splits']
  num_ids_per_split = len(video_ids) // num_splits

  for i in range(num_splits):
    if i != num_splits - 1:
      ids = video_ids[i * num_ids_per_split: (i + 1) * num_ids_per_split]
    else:
      ids = video_ids[i * num_ids_per_split:]

    filename = '%s.%d' % (argv['path_prefix'], i)
    with open(filename, 'w') as fp:
      fp.write('\n'.join(ids))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--video_list', 
      default='data/final_video_id_list.csv', help='')
  parser.add_argument('--num_splits', default=10, help='')
  parser.add_argument('--path_prefix', default='output/video_id_list', help='')

  argv = vars(parser.parse_args())
  print('parsed input parameters:')
  print(json.dumps(argv, indent = 2))

  main(argv)

  print('Done')
