"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import sys
import os

import numpy as np
import tensorflow as tf

import cifar10_input
from model import Model
from pgd_attack import LinfPGDAttack
import argparse

num_classes = 10

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str)
parser.add_argument('--ckpt', type=int)
parser.add_argument('--steps', type=int)
parser.add_argument('--n', type=int)
args = parser.parse_args()

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
model_dir = args.model_dir
data_path = config['data_path']
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(num_classes=num_classes, n=args.n)

# Set up adversary
attack_xent = LinfPGDAttack(model,
                       config['epsilon'],
                       args.steps,
                       config['step_size'],
                       config['random_start'],
                       'xent')

attack_cw = LinfPGDAttack(model,
                       config['epsilon'],
                       args.steps,
                       config['step_size'],
                       config['random_start'],
                       'cw')

saver = tf.train.Saver()

import tqdm

def evaluate(sess):
  total_corr_nat = 0
  total_corr_adv_xent = 0
  total_corr_adv_cw = 0
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  for ibatch in tqdm.tqdm(range(num_batches)):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)

    x_batch = raw_cifar.eval_data.xs[bstart:bend, :]
    y_batch = raw_cifar.eval_data.ys[bstart:bend]
    y_batch = np.eye(num_classes)[y_batch]

    x_batch_adv_xent = attack_xent.perturb(x_batch, y_batch, sess, scaler=1, is_training=False)
    x_batch_adv_cw = attack_cw.perturb(x_batch, y_batch, sess, scaler=1, is_training=False)

    dict_nat = {model.x_input: x_batch,
                model.is_training: False,
                model.y_input: y_batch
                }

    dict_adv_xent = {model.x_input: x_batch_adv_xent,
                model.is_training: False,
                model.y_input: y_batch
                }
    dict_adv_cw = {model.x_input: x_batch_adv_cw,
                model.is_training: False,
                model.y_input: y_batch
                }

    cur_corr_nat = sess.run(model.num_correct,feed_dict = dict_nat)
    cur_corr_adv_xent = sess.run(model.num_correct,feed_dict = dict_adv_xent)
    cur_corr_adv_cw = sess.run(model.num_correct,feed_dict = dict_adv_cw)

    total_corr_nat += cur_corr_nat
    total_corr_adv_xent += cur_corr_adv_xent
    total_corr_adv_cw += cur_corr_adv_cw

  acc_nat = total_corr_nat / num_eval_examples
  acc_adv_xent = total_corr_adv_xent / num_eval_examples
  acc_adv_cw = total_corr_adv_cw / num_eval_examples
  print(str(args.steps) + ' ' + args.model_dir)
  print(f'acc_adv_xent: {acc_adv_xent}')
  print(f'acc_adv_cw: {acc_adv_cw}')
  print(f'acc_nat: {acc_nat}')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # restore
    if args.ckpt == None:
        cur_checkpoint = tf.train.latest_checkpoint(model_dir)
    else:
        cur_checkpoint = os.path.join(args.model_dir, 'checkpoint-%d'%args.ckpt)
    print('@'*20)
    print(cur_checkpoint)
    saver.restore(sess, cur_checkpoint)
    # evaluate
    evaluate(sess)
