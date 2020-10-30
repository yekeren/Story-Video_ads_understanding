import os
import sys
import re
import csv
import json
import numpy as np
from tensorflow import app
from tensorflow import flags
from tensorflow import logging

flags.DEFINE_string('video_dir', '', 
                    'Path to the directory storing videos.')

flags.DEFINE_string('climax_annotation_path', '', 
                    'Path to the annotation file.')

flags.DEFINE_string('output_path', '', 
                    'Path to the output file.')

FLAGS= flags.FLAGS

all_vids = set()
def process_elems(data, vids, annots):
  assert len(vids) == len(annots)
  for vid, annot in zip(vids, annots):
    all_vids.add(vid)
    a1, _, a3 = annot
    if a1 == '3':
      a3 = a3.replace('.', ':').replace(';', ':')
      try:
        mins, secs = a3.split(':')
        secs = int(secs) + int(mins) * 60
        data.setdefault(vid, []).append(secs)
      except Exception as ex:
        print >> sys.stderr, 'We found error: {}, annot={}'.format(ex, a3)

def process_elems_50(data, elems):
  elems = elems[27:]

  vids = elems[:5]
  annots = [
    elems[6:9],
    elems[9:12],
    elems[12:15],
    elems[15:18],
    elems[18:21],
  ]
  process_elems(data, vids, annots)

def process_elems_51(data, elems):
  elems = elems[27:]

  vids = elems[:5]
  annots = [
    elems[7:10],
    elems[10:13],
    elems[13:16],
    elems[16:19],
    elems[19:22],
  ]
  process_elems(data, vids, annots)

def process_elems_48(data, elems):
  elems = elems[27:]

  vids = elems[:5]
  annots = [
    elems[6:9],
    elems[9:12],
    elems[12:15],
    elems[15:18],
    elems[18:21],
  ]
  process_elems(data, vids, annots)

def process_elems_49(data, elems):
  elems = elems[27:]

  vids = elems[:5]
  annots = [
    elems[7:10],
    elems[10:13],
    elems[13:16],
    elems[16:19],
    elems[19:22],
  ]
  process_elems(data, vids, annots)


def main(argv):
  logging.set_verbosity(logging.INFO)

  data = {}
  with open(FLAGS.climax_annotation_path, 'rb') as fp:
    spamreader = csv.reader(fp, delimiter=',', quotechar='"')
    for lineId, elems in enumerate(spamreader):
      if len(elems) == 50:
        process_elems_50(data, elems)
      elif len(elems) == 51:
        process_elems_51(data, elems)
      elif len(elems) == 48:
        process_elems_48(data, elems)
      #elif len(elems) == 49:
      #  process_elems_49(data, elems)
      else:
        assert False, 'line: {}, #elems:{}'.format(lineId, len(elems))
  logging.info('Done')

  with open(FLAGS.output_path, 'w') as fp:
    for vid, annots in data.iteritems():
      fp.write('%s\t%s\n' % (vid, ','.join(map(str, annots))))
  logging.info('total: %i', len(all_vids))

if __name__ == '__main__':
  app.run()
