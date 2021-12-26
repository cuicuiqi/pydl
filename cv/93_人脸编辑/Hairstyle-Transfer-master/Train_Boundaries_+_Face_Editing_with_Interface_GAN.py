#!/usr/bin/env python
# coding: utf-8

# # Settings
# 

# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow as tf
print(tf.version)


# In[ ]:


get_ipython().system('nvcc --version')


# ## InterFaceGAN
# ### Original Repo: https://github.com/ShenYujun/InterFaceGAN
# 

# In[ ]:


get_ipython().system('git clone https://github.com/genforce/interfacegan.git')


# In[ ]:


cd /content/interfacegan/


# In[ ]:


ls


# ## Download the pretrained StyleGAN FFHQ network from NVIDIA:

# In[ ]:


# stylegan ffhq
get_ipython().system('gdown https://drive.google.com/uc?id=1opTWG1jYlyS9TXAuqVyVR68kQWhOhA99')
get_ipython().system('mv /content/interfacegan/karras2019stylegan-ffhq-1024x1024.pkl /content/interfacegan/models/pretrain/karras2019stylegan-ffhq-1024x1024.pkl')


# # Helpers: Train Boundaries

# In[ ]:


# python3.7
"""Utility functions for logging."""

import os
import sys
import logging

__all__ = ['setup_logger']


def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
  """Sets up logger from target work directory.

  The function will sets up a logger with `DEBUG` log level. Two handlers will
  be added to the logger automatically. One is the `sys.stdout` stream, with
  `INFO` log level, which will print improtant messages on the screen. The other
  is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
  be added time stamp and log level before logged.

  NOTE: If `work_dir` or `logfile_name` is empty, the file stream will be
  skipped.

  Args:
    work_dir: The work directory. All intermediate files will be saved here.
      (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

  Returns:
    A `logging.Logger` object.

  Raises:
    SystemExit: If the work directory has already existed, of the logger with
      specified name `logger_name` has already existed.
  """

  logger = logging.getLogger(logger_name)
  if logger.hasHandlers():  # Already existed
    raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                     f'Please use another name, or otherwise the messages '
                     f'may be mixed between these two loggers.')

  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

  # Print log message with `INFO` level or above onto the screen.
  sh = logging.StreamHandler(stream=sys.stdout)
  sh.setLevel(logging.INFO)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  if not work_dir or not logfile_name:
    return logger

  if os.path.exists(work_dir):
    raise SystemExit(f'Work directory `{work_dir}` has already existed!\n'
                     f'Please specify another one.')
  os.makedirs(work_dir)

  # Save log message with all levels in log file.
  fh = logging.FileHandler(os.path.join(work_dir, logfile_name))
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  return logger


# In[ ]:


# python3.7
"""Utility functions for latent codes manipulation."""

import numpy as np
from sklearn import svm

__all__ = ['train_boundary', 'project_boundary', 'linear_interpolate']


def train_boundary(latent_codes,
                   scores,
                   chosen_num_or_ratio=0.02,
                   split_ratio=0.7,
                   invalid_value=None,
                   logger=None):
  """Trains boundary in latent space with offline predicted attribute scores.

  Given a collection of latent codes and the attribute scores predicted from the
  corresponding images, this function will train a linear SVM by treating it as
  a bi-classification problem. Basically, the samples with highest attribute
  scores are treated as positive samples, while those with lowest scores as
  negative. For now, the latent code can ONLY be with 1 dimension.

  NOTE: The returned boundary is with shape (1, latent_space_dim), and also
  normalized with unit norm.

  Args:
    latent_codes: Input latent codes as training data.
    scores: Input attribute scores used to generate training labels.
    chosen_num_or_ratio: How many samples will be chosen as positive (negative)
      samples. If this field lies in range (0, 0.5], `chosen_num_or_ratio *
      latent_codes_num` will be used. Otherwise, `min(chosen_num_or_ratio,
      0.5 * latent_codes_num)` will be used. (default: 0.02)
    split_ratio: Ratio to split training and validation sets. (default: 0.7)
    invalid_value: This field is used to filter out data. (default: None)
    logger: Logger for recording log messages. If set as `None`, a default
      logger, which prints messages from all levels to screen, will be created.
      (default: None)

  Returns:
    A decision boundary with type `numpy.ndarray`.

  Raises:
    ValueError: If the input `latent_codes` or `scores` are with invalid format.
  """
  if not logger:
    logger = setup_logger(work_dir='', logger_name='train_boundary')

  if (not isinstance(latent_codes, np.ndarray) or
      not len(latent_codes.shape) == 2):
    raise ValueError(f'Input `latent_codes` should be with type'
                     f'`numpy.ndarray`, and shape [num_samples, '
                     f'latent_space_dim]!')
  num_samples = latent_codes.shape[0]
  latent_space_dim = latent_codes.shape[1]
  if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
      not scores.shape[0] == num_samples or not scores.shape[1] == 1):
    raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                     f'shape [num_samples, 1], where `num_samples` should be '
                     f'exactly same as that of input `latent_codes`!')
  if chosen_num_or_ratio <= 0:
    raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                     f'but {chosen_num_or_ratio} received!')

  logger.info(f'Filtering training data.')
  if invalid_value is not None:
    latent_codes = latent_codes[scores != invalid_value]
    scores = scores[scores != invalid_value]

  logger.info(f'Sorting scores to get positive and negative samples.')
  sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
  latent_codes = latent_codes[sorted_idx]
  scores = scores[sorted_idx]
  num_samples = latent_codes.shape[0]
  if 0 < chosen_num_or_ratio <= 1:
    chosen_num = int(num_samples * chosen_num_or_ratio)
  else:
    chosen_num = chosen_num_or_ratio
  chosen_num = min(chosen_num, num_samples // 2)

  logger.info(f'Spliting training and validation sets:')
  train_num = int(chosen_num * split_ratio)
  val_num = chosen_num - train_num
  # Positive samples.
  positive_idx = np.arange(chosen_num)
  np.random.shuffle(positive_idx)
  positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
  positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
  # Negative samples.
  negative_idx = np.arange(chosen_num)
  np.random.shuffle(negative_idx)
  negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
  negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
  # import pdb; pdb.set_trace()

  # Training set.
  train_data = np.concatenate([positive_train, negative_train], axis=0)
  train_label = np.concatenate([np.ones(train_num, dtype=np.int),
                                np.zeros(train_num, dtype=np.int)], axis=0)
  logger.info(f'  Training: {train_num} positive, {train_num} negative.')
  # Validation set.
  val_data = np.concatenate([positive_val, negative_val], axis=0)
  val_label = np.concatenate([np.ones(val_num, dtype=np.int),
                              np.zeros(val_num, dtype=np.int)], axis=0)
  logger.info(f'  Validation: {val_num} positive, {val_num} negative.')
  # Remaining set.
  remaining_num = num_samples - chosen_num * 2
  remaining_data = latent_codes[chosen_num:-chosen_num]
  remaining_scores = scores[chosen_num:-chosen_num]
  decision_value = (scores[0] + scores[-1]) / 2
  remaining_label = np.ones(remaining_num, dtype=np.int)
  remaining_label[remaining_scores.ravel() < decision_value] = 0
  remaining_positive_num = np.sum(remaining_label == 1)
  remaining_negative_num = np.sum(remaining_label == 0)
  logger.info(f'  Remaining: {remaining_positive_num} positive, '
              f'{remaining_negative_num} negative.')

  logger.info(f'Training boundary.')
  clf = svm.SVC(kernel='linear')
  
  classifier = clf.fit(train_data, train_label)
  logger.info(f'Finish training.')

  if val_num:
    val_prediction = classifier.predict(val_data)
    correct_num = np.sum(val_label == val_prediction)
    logger.info(f'Accuracy for validation set: '
                f'{correct_num} / {val_num * 2} = '
                f'{correct_num / (val_num * 2):.6f}')

  if remaining_num:
    remaining_prediction = classifier.predict(remaining_data)
    correct_num = np.sum(remaining_label == remaining_prediction)
    logger.info(f'Accuracy for remaining set: '
                f'{correct_num} / {remaining_num} = '
                f'{correct_num / remaining_num:.6f}')

  a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
  return a / np.linalg.norm(a)


def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.

  The function is used for conditional manipulation, where the projected vector
  will be subscribed from the normal direction of the original boundary. Here,
  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].

  NOTE: For now, at most two condition boundaries are supported.

  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.

  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.

  Raises:
    NotImplementedError: If there are more than two condition boundaries.
  """
  if len(args) > 2:
    raise NotImplementedError(f'This function supports projecting with at most '
                              f'two conditions.')
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal
  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)
  if len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)

  raise NotImplementedError


def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
  """Manipulates the given latent code with respect to a particular boundary.

  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.

  Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
  to support W+ space in Style GAN. In this case, all features in W+ space will
  be manipulated same as each other. Accordingly, the output will be with shape
  [10, num_layers, latent_space_dim].

  NOTE: Distance is sign sensitive.

  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                   f'W+ space in Style GAN!\n'
                   f'But {latent_code.shape} is received.')


# # Train Boundaries

# In[ ]:


ls /content


# In[ ]:


get_ipython().system('rm -r boundaries/stylegan_ffhq_straight')


# In[ ]:


# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

# from utils.logger import setup_logger
# from utils.manipulator import train_boundary


def main():
  """Main function."""
  # args = parse_args()
  logger = setup_logger('boundaries/stylegan_ffhq_straight', logger_name='test')

  logger.info('Loading latent codes.')
  latent_codes = np.load("/content/stylegan-dlatents.npy")
  scores = np.load("/content/34_score.npy")

  boundary = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=1,
                            split_ratio=0.7,
                            invalid_value=None,
                            logger=logger)
  np.save(os.path.join('boundaries/stylegan_ffhq_straight', 'boundary.npy'), boundary)


if __name__ == '__main__':
  main()


# In[ ]:


# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

# from utils.logger import setup_logger
# from utils.manipulator import train_boundary


def main():
  """Main function."""
  # args = parse_args()
  logger = setup_logger('boundaries/stylegan_ffhq_wavy', logger_name='wavy')

  logger.info('Loading latent codes.')
  latent_codes = np.load("/content/stylegan-dlatents.npy")
  scores = np.load("/content/3_score.npy")

  boundary = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=1,
                            split_ratio=0.7,
                            invalid_value=None,
                            logger=logger)
  np.save(os.path.join('boundaries/stylegan_ffhq_wavy', 'boundary.npy'), boundary)


if __name__ == '__main__':
  main()


# In[ ]:


# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

# from utils.logger import setup_logger
# from utils.manipulator import train_boundary


def main():
  """Main function."""
  # args = parse_args()
  logger = setup_logger('boundaries/stylegan_ffhq_bangs', logger_name='bangs')

  logger.info('Loading latent codes.')
  latent_codes = np.load("/content/stylegan-dlatents.npy")
  scores = np.load("/content/9_score.npy")

  boundary = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=1,
                            split_ratio=0.7,
                            invalid_value=None,
                            logger=logger)
  np.save(os.path.join('boundaries/stylegan_ffhq_bangs', 'boundary.npy'), boundary)


if __name__ == '__main__':
  main()


# In[ ]:


# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

# from utils.logger import setup_logger
# from utils.manipulator import train_boundary


def main():
  """Main function."""
  # args = parse_args()
  logger = setup_logger('boundaries/stylegan_ffhq_bald', logger_name='bald')

  logger.info('Loading latent codes.')
  latent_codes = np.load("/content/stylegan-dlatents.npy")
  scores = np.load("/content/8_score.npy")

  boundary = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=1,
                            split_ratio=0.7,
                            invalid_value=None,
                            logger=logger)
  np.save(os.path.join('boundaries/stylegan_ffhq_bald', 'boundary.npy'), boundary)


if __name__ == '__main__':
  main()


# In[ ]:


get_ipython().system('mv boundaries/stylegan_ffhq_straight/boundary.npy boundaries/stylegan_ffhq_straight_boundary.npy')
get_ipython().system('mv boundaries/stylegan_ffhq_wavy/boundary.npy boundaries/stylegan_ffhq_wavy_boundary.npy')
get_ipython().system('mv boundaries/stylegan_ffhq_bangs/boundary.npy boundaries/stylegan_ffhq_bangs_boundary.npy')
get_ipython().system('mv boundaries/stylegan_ffhq_bald/boundary.npy boundaries/stylegan_ffhq_bald_boundary.npy')


# # Upoad our latent space vectors 
# 
# Manually upload the output_vectors.npy file 
# 

# In[ ]:


import numpy as np
final_w_vectors = np.load('/content/output_vectors.npy')

print("%d latent vectors of shape %s loaded from %s!" %(final_w_vectors.shape[0], str(final_w_vectors.shape[1:]), 'output_vectors.npy'))


# # II. Let's configure our latent-space interpolation
# ### Change the settings below to morph the faces:

# In[ ]:


latent_direction = 'test'     #### Pick one of ['age', 'eyeglasses', 'gender', 'pose', 'smile']
morph_strength = 3           # Controls how strongly we push the face into a certain latent direction (try 1-5)
nr_interpolation_steps = 48  # The amount of intermediate steps/frames to render along the interpolation path


# # III. Run the latent space manipulation & generate images:

# In[ ]:


# boundary_file = 'stylegan_ffhq_%s_w_boundary.npy' %latent_direction

# print("Ready to start manipulating faces in the ** %s ** direction!" %latent_direction)
# print("Interpolation from %d to %d with %d intermediate frames." %(-morph_strength, morph_strength, nr_interpolation_steps))
# print("\nLoading latent directions from %s" %boundary_file)


# ## Ready? Set, Go!

# In[ ]:


latent_direction = 'wavy'     #### Pick one of ['age', 'eyeglasses', 'gender', 'pose', 'smile']
morph_strength = 3           # Controls how strongly we push the face into a certain latent direction (try 1-5)
nr_interpolation_steps = 48  # The amount of intermediate steps/frames to render along the interpolation path

get_ipython().system('rm -r results/wavy')
get_ipython().system("python edit.py   -m stylegan_ffhq   -b boundaries/stylegan_ffhq_wavy_boundary.npy   -s Wp   -i '/content/output_vectors_four.npy'   -o results/wavy   --start_distance -3.0   --end_distance 3.0   --steps=48")


# In[ ]:


latent_direction = 'straight'     #### Pick one of ['age', 'eyeglasses', 'gender', 'pose', 'smile']
morph_strength = 3           # Controls how strongly we push the face into a certain latent direction (try 1-5)
nr_interpolation_steps = 48  # The amount of intermediate steps/frames to render along the interpolation path

get_ipython().system('rm -r results/straight')
get_ipython().system("python edit.py   -m stylegan_ffhq   -b boundaries/stylegan_ffhq_straight_boundary.npy   -s Wp   -i '/content/output_vectors_four.npy'   -o results/straight   --start_distance -3.0   --end_distance 3.0   --steps=48")


# In[ ]:


latent_direction = 'bangs'     #### Pick one of ['age', 'eyeglasses', 'gender', 'pose', 'smile']
morph_strength = 3           # Controls how strongly we push the face into a certain latent direction (try 1-5)
nr_interpolation_steps = 48  # The amount of intermediate steps/frames to render along the interpolation path

get_ipython().system('rm -r results/bangs')
get_ipython().system("python edit.py   -m stylegan_ffhq   -b boundaries/stylegan_ffhq_bangs_boundary.npy   -s Wp   -i '/content/output_vectors_four.npy'   -o results/bangs   --start_distance -3.0   --end_distance 3.0   --steps=48")


# In[ ]:


latent_direction = 'bald'     #### Pick one of ['age', 'eyeglasses', 'gender', 'pose', 'smile']
morph_strength = 3           # Controls how strongly we push the face into a certain latent direction (try 1-5)
nr_interpolation_steps = 48  # The amount of intermediate steps/frames to render along the interpolation path

get_ipython().system('rm -r results/bald')
get_ipython().system("python edit.py   -m stylegan_ffhq   -b boundaries/stylegan_ffhq_bald_boundary.npy   -s Wp   -i '/content/output_vectors_four.npy'   -o results/bald   --start_distance -3.0   --end_distance 3.0   --steps=48")


# In[ ]:


# latent_direction = 'bald'     #### Pick one of ['age', 'eyeglasses', 'gender', 'pose', 'smile']
# morph_strength = 3           # Controls how strongly we push the face into a certain latent direction (try 1-5)
# nr_interpolation_steps = 48  # The amount of intermediate steps/frames to render along the interpolation path

# !rm -r results/bald
# !python edit.py \
#   -m stylegan_ffhq \
#   -b boundary_bald_without_smile.npy \
#   -s Wp \
#   -i '/content/output_vectors.npy' \
#   -o results/bald \
#   --start_distance -3.0 \
#   --end_distance 3.0 \
#   --steps=48


# # IV. Finally, turn the results into pretty movies!
# Adjust which video to render & at what framerate:

# In[ ]:


# latent_direction = 'bangs'
latent_direction = 'bald'
# latent_direction = 'straight'
# latent_direction = 'wavy'

image_folder = '/content/interfacegan/results/%s' %latent_direction
video_fps = 12.


# ### Render the videos:

# In[ ]:


from moviepy.editor import *
import cv2

out_path = '/content/output_videos/'

images = [img_path for img_path in sorted(os.listdir(image_folder)) if '.jpg' in img_path]
os.makedirs(out_path, exist_ok=True)

prev_id = None
img_sets = []
for img_path in images:
  img_id = img_path.split('_')[0]
  if img_id == prev_id: #append
    img_sets[-1].append(img_path)
    
  else: #start a new img set
    img_sets.append([])
    img_sets[-1].append(img_path)
  prev_id = img_id

print("Found %d image sets!\n" %len(img_sets))
if image_folder[-1] != '/':
  image_folder += '/'

def make_video(images, vid_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(vid_name, fourcc, video_fps, (1024, 1024))
    gen = {}
    for img in images:
      video.write(img)
    video.release()
    print('finished '+ vid_name)
    
    
for i in range(len(img_sets)):
  print("############################")
  print("\nGenerating video %d..." %i)
  set_images = []
  vid_name = out_path + 'out_video_%s_%02d.mp4' %(latent_direction,i)
  for img_path in img_sets[i]:
    set_images.append(cv2.imread(image_folder + img_path))

  set_images.extend(reversed(set_images))
  make_video(set_images, vid_name)


# # Results

# In[ ]:


video_file_to_show = 0
# latent_direction = 'wavy'

clip = VideoFileClip('/content/output_videos/out_video_%s_%02d.mp4' %(latent_direction, video_file_to_show))
clip.ipython_display(height=512, autoplay=1, loop=1)


# In[ ]:


video_file_to_show = 1
# latent_direction = 'bald'

clip = VideoFileClip('/content/output_videos/out_video_%s_%02d.mp4' %(latent_direction, video_file_to_show))
clip.ipython_display(height=512, autoplay=1, loop=1)


# In[ ]:


video_file_to_show = 2

clip = VideoFileClip('/content/output_videos/out_video_%s_%02d.mp4' %(latent_direction, video_file_to_show))
clip.ipython_display(height=512, autoplay=1, loop=1)


# In[ ]:


video_file_to_show = 3

clip = VideoFileClip('/content/output_videos/out_video_%s_%02d.mp4' %(latent_direction, video_file_to_show))
clip.ipython_display(height=512, autoplay=1, loop=1)


# In[ ]:


video_file_to_show = 4

clip = VideoFileClip('/content/output_videos/out_video_%s_%02d.mp4' %(latent_direction, video_file_to_show))
clip.ipython_display(height=512, autoplay=1, loop=1)


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


get_ipython().system("cp -r /content/interfacegan/results/straight '/content/gdrive/My Drive/hairsyle-images/34-straight/four'")


# In[ ]:


get_ipython().system("cp -r /content/interfacegan/results/wavy '/content/gdrive/My Drive/hairsyle-images/3-wavy/four'")


# In[ ]:


get_ipython().system("cp -r /content/interfacegan/results/bald '/content/gdrive/My Drive/hairsyle-images/8-bald/four'")


# In[ ]:


get_ipython().system("cp -r /content/interfacegan/results/bangs/ '/content/gdrive/My Drive/hairsyle-images/9-bangs/four'")


# In[ ]:


get_ipython().system("cp -r /content/output_videos/ '/content/gdrive/My Drive/hairsyle-images/'")


# # conditional 
# 

# In[ ]:


def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.
  The function is used for conditional manipulation, where the projected vector
  will be subscribed from the normal direction of the original boundary. Here,
  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].
  NOTE: For now, at most two condition boundaries are supported.
  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.
  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.
  Raises:
    NotImplementedError: If there are more than two condition boundaries.
  """
  if len(args) > 2:
    raise NotImplementedError(f'This function supports projecting with at most '
                              f'two conditions.')
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal
  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)
  if len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)

  raise NotImplementedError


# In[ ]:


ls boundaries/


# In[ ]:


import numpy as np

primal = np.load('boundaries/stylegan_ffhq_test_boundary_8.npy') 
condition = np.load('boundaries/stylegan_ffhq_smile_w_boundary.npy') 
conditional_primal = project_boundary(primal, condition)
np.save('boundary_bald_without_smile.npy',conditional_primal)


# # Experiment
# 
# * You can blend between two faces by doing a linear interpolation in the latent space: very cool!
# *   The StyleGAN vector has 18x512 dimensions, each of those 18 going into a different layer of the generator...
# *   You could eg take the first 9 from person A and the next 9 from person B
# *   This is why it's called "Style-GAN": you can manipulate the style of an image at multiple levels of the Generator!
# *   Try interpolating in Z-space rather than in W-space (see InterFaceGan paper & repo)
