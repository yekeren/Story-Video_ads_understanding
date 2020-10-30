# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import re
import cv2
import numpy as np
import json
from PIL import Image
from tensorflow import flags
from tensorflow import logging

flags.DEFINE_string('video_dir', '', 
                    'Path to the directory storing videos.')

flags.DEFINE_string('video_id_path', '', 
                    'Path to the file which stores video ids.')

flags.DEFINE_string('output_dir', '', 
                    'Path to the output directory.')

flags.DEFINE_integer('batch_size', 1, '')

FLAGS= flags.FLAGS

logging.set_verbosity(logging.INFO)
def _video_id_iterator(filename):
  with open(filename, 'r') as fp:
    return [line.strip('\n') for line in fp.readlines()]


def _get_screenshots(dirname):
  filenames = filter(
      lambda x: re.match(r'screenshot\d+.jpg', x), 
      os.listdir(dirname))
  filenames = map(
      lambda x: (re.match(r'screenshot(\d+).jpg', x).group(1), x), 
      filenames)

  filenames.sort(lambda x, y: cmp(int(x[0]), int(y[0])))
  filenames = [filename[1] for filename in filenames]
  return filenames

def _read_image(filename):
  """Reads image data from file.

  Args:
    filename: the path to the image file.
  """
  bgr = cv2.imread(filename)
  rgb = bgr[:, :, ::-1]
  return rgb

def _extract_video_feature(video_id, video_dir, output_dir, extract_fn):
  """Extracts features from video.
  """
  dirname = os.path.join(video_dir, '%s' % video_id)

  filenames = _get_screenshots(dirname)
  n_frames = len(filenames)
  batch_size = FLAGS.batch_size
  assert batch_size == 1

  # Batch process the frames.
  features, detections, batch = [], [], []
  for index , filename in enumerate(filenames):
    image_data = Image.open(os.path.join(dirname, filename))
    predictions, det_results = extract_fn(video_id, image_data)
    features.append(predictions)
    detections.extend(det_results)

  features = np.concatenate(features, axis=0)
  assert features.shape[0] == len(filenames)
  assert len(detections) == len(filenames)

  # Write results to output path.
  filename = os.path.join(output_dir, '%s.npz' % video_id)
  with open(filename, 'wb') as fp:
    np.save(fp, features)

  filename = os.path.join(output_dir, '%s.json' % video_id)
  with open(filename, 'w') as fp:
    fp.write(json.dumps(detections))

  logging.info('Video features of %s are saved to %s.', video_id, filename)



# th architecture to use
arch = 'resnet50'

# load the pre-trained weights
model_file = 'places365/whole_%s_places365_python36.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

useGPU = 1
if useGPU == 1:
    model = torch.load(model_file)
    model.cuda()
else:
    model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

## assume all the script in python36, so the following is not necessary
## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
#from functools import partial
#import pickle
#pickle.load = partial(pickle.load, encoding="latin1")
#pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
#torch.save(model, 'whole_%s_places365_python36.pth.tar'%arch)

model.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the test image
img_name = '12.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    os.system('wget ' + img_url)

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0), volatile=True).cuda()

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

print('RESULT ON ' + img_name)
# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

def _extract_feature(video_id, img):
  # Transpose to [batch, channels, width, height]
  input_img = V(centre_crop(img).unsqueeze(0), volatile=True).cuda()

  logit = model.forward(input_img)
  h_x = F.softmax(logit, 1)
  features = h_x.data.cpu().numpy()

  h_x = h_x.data.squeeze()
  probs, idx = h_x.sort(0, True)

  # output the prediction
  detection_result = []
  for i in xrange(5):
    #print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    detection_result.append({
        'cid': idx[i],
        'cname': classes[idx[i]],
        'score': round(probs[i], 3),
        'bounding_box': {'x1': 0.0, 'y1': 0.0, 'x2': 1.0, 'y2': 1.0}
        })
  det_results = [detection_result]

  return features, det_results

# Iterate through video ids.
for video_id in _video_id_iterator(FLAGS.video_id_path):
  _extract_video_feature(video_id, FLAGS.video_dir, 
      FLAGS.output_dir, _extract_feature)

logging.info('Done')
