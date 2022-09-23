"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
import sys
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
import math

import cifar10_input
import cifar100_input
import imagenet32_input
from pgd_attack import LinfPGDAttack
import logging
import argparse
from model_biamat import Model

parser = argparse.ArgumentParser()
parser.add_argument('--primary', default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--auxiliary', default='imagenet', choices=['imagenet'])
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--suffix', type=str, help='suffix')
parser.add_argument('--n', type=int)
parser.add_argument('--warmup-epoch', type=int, default=5)
parser.add_argument('--alpha', type=float, help='coefficient for aux loss')
parser.add_argument('--gamma', type=float, help='a hyperparameter for biamat gate')
args = parser.parse_args()

conf = 'config.json'
with open(conf) as config_file:
    config = json.load(config_file)

model_dir = config['model_dir'] + args.suffix
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Args: %s', args)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path_imagenet = config['data_path_imagenet']
momentum = config['momentum']
batch_size = config['training_batch_size']
eval_batch_size = config['eval_batch_size']

howmany = batch_size // 2
'''''''''
Primary dataset
'''''''''
if args.primary == 'cifar10':
    num_classes = 10
    raw_cifar = cifar10_input.CIFAR10Data(config['data_path_10'])
elif args.primary == 'cifar100':
    num_classes = 100
    raw_cifar = cifar100_input.CIFAR100Data(config['data_path_100'])
else:
    logging.info("add a dataloader for the primary dataset")
    sys.exit(1)
num_eval_examples = raw_cifar.eval_data.n

'''''''''
Auxiliary dataset
'''''''''
if args.auxiliary == 'imagenet':
    aux_num_classes = 1000
    raw_data = imagenet32_input.ImageNetData(data_path_imagenet)
else:
    logging.info("add a dataloader for the auxiliary dataset")
    sys.exit(1)

# Setting up the data and the model
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(num_classes=num_classes, aux_num_classes=aux_num_classes, n=args.n)

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent_merge + weight_decay * model.weight_decay_loss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
        total_loss,
        global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       'merge')

attack_eval = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

saver = tf.train.Saver(max_to_keep=5)
tf.summary.scalar('primary_accuracy_adv_train', model.accuracy)
tf.summary.scalar('auxiliary_accuracy_adv_train', model.accuracy_aux)
tf.summary.scalar('primary_xent_adv_train', model.xent_pri)
tf.summary.scalar('auxiliary_xent_adv_train', model.xent_aux)
#tf.summary.image('images adv train', model.x_input, max_outputs=128)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy(conf, model_dir)

tops = [0]
max_acc_adv = 0
acc_nat_at_max_adv = 0
def comp_and_save_ckpt(tops, new_adv, new_nat, summary_writer, sess, global_step):
  m = min(tops)
  if m <= new_adv:
    tops.remove(m)
    tops.append(new_adv)
    saver.save(sess,
               os.path.join(model_dir, 'checkpoint'),
               global_step=global_step)
    summary_comp = tf.Summary(value=[
          tf.Summary.Value(tag='max acc adv', simple_value= max(tops)),
          tf.Summary.Value(tag='acc nat at max adv', simple_value= new_nat)])
    summary_writer.add_summary(summary_comp, global_step.eval(sess))
  return tops

def evaluate(sess):
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr_nat = 0
  total_corr_adv = 0
  for ibatch in range(num_batches):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)

    x_batch = raw_cifar.eval_data.xs[bstart:bend, :]
    y_batch = raw_cifar.eval_data.ys[bstart:bend]
    y_batch = np.eye(num_classes)[y_batch]

    x_batch_adv = attack_eval.perturb(x_batch, y_batch, sess, scaler=1.0, is_training=False)

    dict_nat = {model.x_input: x_batch,
                model.is_training: False,
                model.y_input: y_batch
                }

    dict_adv = {model.x_input: x_batch_adv,
                model.is_training: False,
                model.y_input: y_batch
                }

    cur_corr_nat = sess.run(model.num_correct,feed_dict = dict_nat)
    cur_corr_adv = sess.run(model.num_correct,feed_dict = dict_adv)

    total_corr_nat += cur_corr_nat
    total_corr_adv += cur_corr_adv

  acc_nat = total_corr_nat / num_eval_examples
  acc_adv = total_corr_adv / num_eval_examples
  logging.info('Evaluation:\t acc adv: {:.6f}, acc nat: {:.6f}'.format(acc_adv, acc_nat))

  summary_eval = tf.Summary(value=[
        tf.Summary.Value(tag='acc nat', simple_value= acc_nat),
        tf.Summary.Value(tag='acc adv', simple_value= acc_adv)])
  return summary_eval, acc_adv, acc_nat

def biamat_gate(sess, ii, x, y, x_aux, y_aux, th):
    now_epoch = ii // (raw_cifar.train_data.n // howmany)
    num_pri = len(y)
    if now_epoch < args.warmup_epoch:
        return np.concatenate([x, x_aux], axis=0), np.eye(num_classes)[y], y_aux, 0, 0
    if th == 0:
        d = {model.x_input: np.concatenate([x, x_aux], axis=0),
             model.is_training: False}
        confidence = sess.run(model.confidence, feed_dict = d)
        confi_pri = confidence[:num_pri]
        mean_confi_pri = np.mean(confi_pri)
        confi_aux = confidence[num_pri:]
        threshold = args.gamma*mean_confi_pri 
    else:
        d = {model.x_input: x_aux,
             model.is_training: False}
        confi_aux = sess.run(model.confidence, feed_dict = d)
        threshold = th
    mask = confi_aux < threshold
    num_aux_out = np.sum(mask)
    x_aux_out = x_aux[mask]
    x_aux_in = x_aux[np.logical_not(mask)]
    x_batch = np.concatenate([x, x_aux_out, x_aux_in], axis=0)
    y_batch_random = np.ones([num_aux_out, num_classes]) * (1./num_classes)
    y_batch = np.concatenate([np.eye(num_classes)[y], y_batch_random], axis=0)
    y_aux_batch = y_aux[np.logical_not(mask)]
    return x_batch, y_batch, y_aux_batch, threshold, args.alpha
    

with tf.Session() as sess:
  # initialize data augmentation
  if args.primary == 'cifar10':
      cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
  elif args.primary == 'cifar100':
      cifar = cifar100_input.AugmentedCIFAR100Data(raw_cifar, sess, model)
  if args.auxiliary == 'imagenet':
      data = imagenet32_input.AugmentedImageNetData(raw_data, sess, model)
  if args.shuffle:
      args.warmup_epoch = 0
      args.gamma = 100
  sess.run(tf.global_variables_initializer())
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  training_time = 0.0
  # Main training loop
  biamat = 0 # n_high / n_aux
  th = 0
  for ii in range(max_num_training_steps):
    x_batch10, y_batch10 = cifar.train_data.get_next_batch(batch_size - howmany,
                                                       multiple_passes=True)
    x_batch_aug, y_batch_aug = data.train_data.get_next_batch(howmany, multiple_passes=True)

    x_batch, y_batch, y_aux_batch, th, alpha = biamat_gate(sess, ii, x_batch10, y_batch10, x_batch_aug, y_batch_aug, th)
    biamat += len(y_aux_batch) / float(len(y_batch_aug))

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb_pri_and_aux(x_batch, y_batch, y_aux_batch, sess, is_training=True)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.is_training: True,
                model.num_aux_out: len(y_batch_aug) - len(y_aux_batch),
                model.num_aux_in: len(y_aux_batch),
                model.y_input_aux: y_aux_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.is_training: True,
                model.scaler: alpha,
                model.num_aux_out: len(y_batch_aug) - len(y_aux_batch),
                model.num_aux_in: len(y_aux_batch),
                model.y_input_aux: y_aux_batch,
                model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc, loss_pri, loss_aux = sess.run([model.accuracy, model.xent_pri, model.xent_aux], feed_dict=adv_dict)
      logging.info('Step {}:    ({})'.format(ii, datetime.now()))
      logging.info('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      logging.info('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      logging.info('    loss primary {:.4}'.format(loss_pri))
      logging.info('    loss auxiliary {:.4}'.format(loss_aux))
      logging.info('    biamat ratio {:.4}%'.format(biamat * 100 / num_output_steps))
      summary_biamat_ratio = tf.Summary(value=[
            tf.Summary.Value(tag='biamat ratio', simple_value=biamat * 100 / num_output_steps)])
      summary_writer.add_summary(summary_biamat_ratio, global_step.eval(sess))
      biamat = 0
      if ii != 0:
        logging.info('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
      logging.info(120 * '=')

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start

    # Tensorboard summaries
    if (ii == 0) or ((ii+1) % num_summary_steps == 0):
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))
      summary_eval, acc_adv, acc_nat = evaluate(sess)
      summary_writer.add_summary(summary_eval, global_step.eval(sess))
      if ii > step_size_schedule[1][0]:
        tops = comp_and_save_ckpt(tops, acc_adv, acc_nat, summary_writer, sess, global_step)

  #saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
