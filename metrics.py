"""Script to run precision and recall when two image folders (real/fake folder) is provided."""

import os
import sys
import argparse
import tensorflow as tf
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
tf.logging.set_verbosity(tf.logging.ERROR)

import dnnlib
from dnnlib.util import Logger
from utils import init_tf

import pathlib
import random
import numpy as np
from PIL import Image
from time import time

from precision_recall import DistanceBlock
from precision_recall import knn_precision_recall_features
from precision_recall import ManifoldEstimator
from utils import initialize_feature_extractor


# Minimal CLI.

def parse_command_line_arguments(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Improved Precision and Recall Metric for Assessing Generative Models.',
                                     epilog='This script can be used to reproduce StyleGAN truncation sweep (Fig. 4) and' \
                                            ' computing realism score for StyleGAN samples (Fig. 11).')

    parser.add_argument(
        '-r',
        '--path_real',
        type=str,
        required=True,
        help='Absolute path to real directory.'
    )
    parser.add_argument(
        '-f',
        '--path_fake',
        type=str,
        required=True,
        help='Absolute path to fake directory.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=50,
        help='batch size.'
    )
    parser.add_argument(
        '-n',
        '--num_images',
        type=int,
        default=50000,
        help='number of input images.'
    )
    parsed_args, _ = parser.parse_known_args(args)
    return parsed_args

#----------------------------------------------------------------------------

def main(args=None):
    # Parse command line arguments.
    opt = parse_command_line_arguments(args)

    compute_precision_recall(opt.path_real, opt.path_fake, opt.batch_size, opt.num_images,
                             num_gpus=1, save_txt=None, save_path=None)

#     peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
#     peak_gpu_mem_usage = peak_gpu_mem_op.eval()
#     print('Peak GPU memory usage: %g GB' % (peak_gpu_mem_usage * 1e-9))

#----------------------------------------------------------------------------

def compute_precision_recall(path_real, path_fake, minibatch_size, num_images,
                             num_gpus=1, save_txt=None, save_path=None):
    """StyleGAN truncation sweep. (Fig. 4)

        Args:
            datareader (): FFHQ datareader object.
            minibatch_size (int): Minibatch size.
            num_images (int): Number of images used to evaluate precision and recall.
            truncations (list): List of truncation psi values.
            save_txt (string): Name of result file.
            save_path (string): Absolute path to directory where result textfile is saved.
            num_gpus (int): Number of GPUs used.
            random_seed (int): Random seed.

    """
    init_tf()
    
    # Initialize VGG-16.
    feature_net = initialize_feature_extractor()

    it_start = time()

    # Calculate VGG-16 features for real images.
    print('Reading real images...')
    ref_features = get_features(path_real, feature_net, num_images, minibatch_size, num_gpus=num_gpus)

    # Calculate VGG-16 features for generated images.
    print('Generating images...')
    eval_features = get_features(path_fake, feature_net, num_images, minibatch_size, num_gpus=num_gpus)

    # Calculate k-NN precision and recall.
    state = knn_precision_recall_features(ref_features, eval_features, num_gpus=num_gpus)

    # Store results.
    precision = state['precision'][0]
    recall = state['recall'][0]

    # Print progress.
    print('Precision: %0.3f' % precision)
    print('Recall: %0.3f' % recall)
    print('Run time: %gs\n' % (time() - it_start))

    # Save results.
    if save_txt:
        result_path = save_path
        result_file = os.path.join(result_path, 'stylegan_truncation.txt')
        with open(result_file, 'w') as f:
            f.write(f"Precision: {precision}\nRecall: {recall}\n")
    
    return precision, recall


def get_features(images, feature_net, num_images, minibatch_size, num_gpus=1):
    if type(images) == str:
        input_images = get_all_images(images, num_images)
    elif type(images) == list:
        input_images = [im.transpose((2, 0, 1)) for im in images[:num_images]]
    else:
        raise ValueError(f"type of images should be list or string, but got {type(images)}")

    assert num_images == input_images.shape[0]
    features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)

    for begin in range(0, num_images, minibatch_size):
        end = min(begin + minibatch_size, num_images)
        real_batch = input_images[begin:end]
        features[begin:end] = feature_net.run(real_batch, num_gpus=num_gpus, assume_frozen=True)
    
    return features


def get_all_images(path, num_images):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

    assert num_images <= len(files)
    if num_images < len(files):
        print("dataset size is larger than required!!")
        files = random.sample(files, num_images)
    
    assert num_images == len(files)
    x = []
    for fn in files:
        im = Image.open(fn).convert('RGB')
        x.append(np.asarray(im).transpose((2, 0, 1)))
    x = np.array(x)
    return x


if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
