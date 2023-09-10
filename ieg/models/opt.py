# coding=utf-8
"""Fast Sample Reweighting With Optimization."""

import gc
import os

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from ieg import utils
from ieg.models import networks
from ieg.models.custom_ops import logit_norm
from ieg.models.custom_ops import MixMode
from ieg.models.l2rmodel import L2R
from ieg.LP import LinearProgramming

FLAGS = flags.FLAGS
logging = tf.logging


def reduce_mean(vectors):
  """Reduces mean without nan."""
  return tf.where(
      tf.size(vectors) > 0, tf.reduce_mean(vectors),
      tf.zeros((), dtype=vectors.dtype))


def softmax(q, axis=-1):
  exps = np.exp(q - np.max(q, axis=-1, keepdims=True))
  return exps / np.sum(exps, axis=axis, keepdims=True)


class Queue(object):
  """Queue."""

  def __init__(
      self,
      sess,
      dataset_ds,
      dataset_size,
      nclass,
      shape=(32, 32, 3),
      capacity=1000,
      batch_size=200,
      beta=0.0,  # use [0, 1] to smooth past history information
      metric='loss'):  # single gpu default
    self.init_capacity = capacity
    self.capacity = self.init_capacity
    self.sess = sess
    self.nclass = nclass
    self.dataset_size = dataset_size
    self.metric = metric
    self.batch_size = batch_size
    self.beta = beta
    self.summaries = []
    self.shape = shape
    assert capacity % (2 * nclass) == 0
    assert batch_size % nclass == 0
    assert capacity >= nclass, 'Class number larger than capacity'

    with tf.device('/cpu:0'):
      with tf.variable_scope('queue'):
        self.queue_probe_images = tf.Variable(
            tf.zeros(shape=[batch_size] + list(shape), dtype=tf.float32),
            trainable=False,
            name='probe_images',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.queue_probe_labels = tf.Variable(
            tf.zeros(shape=[
                batch_size,
            ], dtype=tf.int32),
            trainable=False,
            name='probe_labels',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.queue_probe_lam = tf.Variable(
            tf.zeros(shape=[
                batch_size,
            ], dtype=tf.float32),
            trainable=False,
            name='probe_lam',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.extract_images = tf.Variable(
            tf.zeros(shape=[capacity] + list(shape), dtype=tf.float32),
            trainable=False,
            name='extract_images',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.extract_labels = tf.Variable(
            tf.zeros(shape=[capacity,], dtype=tf.int32),
            trainable=False,
            name='extract_labels',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        #if FLAGS.use_pseudo_loss:
        self.queue_ema_logits = tf.Variable(
            tf.zeros(shape=[dataset_size, nclass], dtype=tf.float32),
            trainable=False,
            name='ema_logits',
            aggregation=tf.VariableAggregation.MEAN)

        self.queue_all_weights = tf.Variable(
              tf.zeros((dataset_size, ), dtype=tf.float32),
              trainable=False,
              name='ema_weights',
              aggregation=tf.VariableAggregation.MEAN)
        self.queue_all_save_labels = tf.Variable(
              tf.zeros((dataset_size, ), dtype=tf.int32),
              trainable=False,
              name='ema_save_labels',
              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.queue_all_golden_labels = tf.Variable(
              tf.zeros((dataset_size, ), dtype=tf.int32),
              trainable=False,
              name='ema_golden_labels',
              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        self.queue_all_loss = tf.Variable(
              tf.zeros((dataset_size, ), dtype=tf.float32),
              trainable=False,
              name='ema_loss',
              aggregation=tf.VariableAggregation.MEAN)
        self.queue_all_meta_loss = tf.Variable(
              tf.zeros((dataset_size, ), dtype=tf.float32),
              trainable=False,
              name='ema_meta_loss',
              aggregation=tf.VariableAggregation.MEAN)
        self.queue_all_loss_margin = tf.Variable(
              tf.zeros((dataset_size, ), dtype=tf.float32),
              trainable=False,
              name='ema_loss_margin',
              aggregation=tf.VariableAggregation.MEAN)

        self.queue_data_indices = tf.Variable(
            tf.zeros((self.init_capacity,), dtype=tf.float32),
            trainable=False,
            name='indices',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        # For monitors
        self.purity_log = tf.Variable(
            tf.zeros((), dtype=tf.float32),
            trainable=False,
            name='purity',
            aggregation=tf.VariableAggregation.MEAN)
        self.capacity_log = tf.Variable(
            tf.zeros((), dtype=tf.float32),
            trainable=False,
            name='capacity',
            aggregation=tf.VariableAggregation.MEAN)

    self.summaries.append(
        tf.summary.histogram('queue/indices', self.queue_data_indices))
    self.summaries.append(tf.summary.scalar('queue/purity', self.purity_log))
    self.summaries.append(
        tf.summary.scalar('queue/capacity', self.capacity_log))
    self.summaries.append(
        tf.summary.histogram('queue/labels', self.queue_probe_labels))

    self.plh_probe_images = tf.placeholder(tf.float32, shape=[batch_size] + list(shape))
    self.plh_probe_labels = tf.placeholder(tf.int32, shape=[batch_size,])
    self.plh_probe_lam = tf.placeholder(tf.float32, shape=[batch_size,])
    self.plh_single_probe_images = tf.placeholder(tf.float32, shape=[1] + list(shape))
    self.plh_single_probe_labels = tf.placeholder(tf.int32, shape=[1,])
    self.plh_extract_images = tf.placeholder(tf.float32, shape=[capacity] + list(shape))
    self.plh_extract_labels = tf.placeholder(tf.int32, shape=[capacity,])
    # There are three global numpy variables to maintain.
    # ds_pre_scores is used for forget_event metric.
    self.ds_pre_scores = np.zeros((dataset_size, 1), np.float32).squeeze()
    self.ds_rates = np.zeros((dataset_size, 1), np.float32).squeeze()
    # initialize as -1 for sanity check
    self.ds_labels = np.zeros((dataset_size,), np.int32) - 1
    self.global_gold_labels = np.zeros((dataset_size,), np.int32) - 1

    self.dataset_ds = dataset_ds.batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).apply(
            tf.data.experimental.ignore_errors())

    self.queue_all_images = np.zeros([batch_size] + list(shape), np.float32)
    # Init to be -1.
    self.queue_all_labels = np.zeros([batch_size,], np.float32) - 1
    self.queue_all_ids = np.zeros([batch_size,], np.int32) - 1

    self.queue_all_extract_images = np.zeros([capacity] + list(shape), np.float32)
    self.queue_all_extract_labels = np.zeros([capacity,], np.float32) - 1

    self.ids = []
    np.random.seed(FLAGS.seed)

  def save(self, path, iter):
    """Saves queue info."""
    with tf.gfile.Open(os.path.join(path, f'queue_info-{iter}.npy'), 'w') as fout:
      with self.sess.as_default():
        np.save(
            fout, {
                'ds_labels': self.ds_labels,
                'ds_rates': self.ds_rates,
                'ds_pre_scores': self.ds_pre_scores,
                'queue_all_golden_labels': self.queue_all_golden_labels.eval(),
                'queue_all_save_labels': self.queue_all_save_labels.eval(),
                'queue_all_weights': self.queue_all_weights.eval(),
                'queue_ema_logits': self.queue_ema_logits.eval(),
                'queue_all_loss': self.queue_all_loss.eval(),
                'queue_all_meta_loss': self.queue_all_meta_loss.eval(),
                'queue_all_loss_margin': self.queue_all_loss_margin.eval(),
            },
            allow_pickle=True)

  def load(self, path, iter):
    """Loads from latest checkpoints."""
    with tf.gfile.Open(os.path.join(path, f'queue_info-{iter}.npy'), 'rb') as fin:
      data = np.load(fin, allow_pickle=True)
      self.ds_labels = data.item().get('ds_labels')
      self.ds_rates = data.item().get('ds_rates')
      self.ds_pre_scores = data.item().get('ds_pre_scores')
      self.global_gold_labels = data.item().get('queue_all_golden_labels')

  def _cls_bal_sampling(self, labels, total):
    """Class-balanced subsampling total data from according to labels."""
    ulbs = np.unique(labels)
    assert len(ulbs) == self.nclass
    assert total % len(ulbs) == 0
    npc = total // len(ulbs)
    candidates = []
    for i in ulbs:
      cad_ids = np.where(labels == i)[0]
      try:
        candidates.append(np.random.choice(cad_ids, size=npc, replace=False))
      except:   # pylint: disable=bare-except
        logging.warning('Class {} has only {} samples (requested {})'.format(
            i, len(cad_ids), npc))
        candidates.append(np.random.choice(cad_ids, size=npc))
    return np.concatenate(candidates)

  def get_lam(self, R):
    lam = np.zeros((R.shape[0]), dtype=np.float32)
    # lam = np.ones((R.shape[0]), dtype=np.float32) / R.shape[0]
    gt = np.zeros(self.capacity)
    if FLAGS.clean_opt:
      pos_idx = np.where(self.constraint_is_correct)[0]
      neg_idx = np.where(~self.constraint_is_correct)[0]
    else:
      pos_idx = np.arange(self.capacity // 2)
      neg_idx = np.arange(self.capacity // 2, self.capacity)
    gt[pos_idx] = 1
    LP = LinearProgramming(R, gt, lam, self.queue_all_extract_labels, pos_idx, neg_idx, (1,1,1), 1)
    LP.optimize(.1,freq=50)
    lam = LP.lam.detach().numpy()
    lam = lam / lam.sum()
    return lam

  def _update_scores(self, *args):
    """Update scores for data points in ids."""
    labels, ids, logits = args[:3]  # pylint: disable=unbalanced-tuple-unpacking
    # Meta-margin
    loss_margin = args[-1]
    updates = score = -loss_margin


    if self.beta >= 1:
      self.ds_rates[ids] += self.beta * updates
    else:
      self.ds_rates[ids] = self.beta * self.ds_rates[ids] + (
          1 - self.beta) * updates
    self.ds_pre_scores[ids] = score
    self.ds_labels[ids] = labels
    self.ids.extend(ids.tolist())

  def _get_top_indices(self, n_large, n_small):
    """Gets top best scored data."""

    global_ds_ids = np.arange(self.ds_rates.shape[0])
    return_ids = []
    for i in range(self.nclass):
      loc_ds_ids = global_ds_ids[self.ds_labels == i]
      loc_pre_rate = self.ds_rates[self.ds_labels == i]
      # From low to high, lower is cleaner (better) data to be selected in queue
      loc_ranked_inds = np.random.choice(np.argsort(loc_pre_rate)[:n_large], n_small, replace=False)
      return_ids.append(loc_ds_ids[loc_ranked_inds])

    return_ids = np.concatenate(return_ids)
    return return_ids

  def _get_top_or_bottom_indices(self, topk):
    """Gets top best scored data."""

    global_ds_ids = np.arange(self.ds_rates.shape[0])
    return_ids = []
    for i in range(self.nclass):
      # if FLAGS.clean_opt:
      #   loc_ds_ids = global_ds_ids[(self.ds_labels == i) & (self.global_gold_labels == i)]
      # else:
      #   loc_ds_ids = global_ds_ids[self.ds_labels == i]
      loc_ds_ids = global_ds_ids[self.ds_labels == i]
      loc_pre_rate = self.ds_rates[loc_ds_ids]
      # From low to high, lower is cleaner (better) data to be selected in queue
      if topk > 0:
        loc_ranked_inds = np.argsort(loc_pre_rate)[:topk]
      else:
        loc_ranked_inds = np.argsort(loc_pre_rate)[topk:]
      return_ids.append(loc_ds_ids[loc_ranked_inds])

    return_ids = np.concatenate(return_ids)
    return return_ids

  def _enqueue(self, imgs, labels, ids):
    if len(imgs.shape) == 5:
      # The 2nd dim has one standard and one strong augmentation
      rid = np.random.randint(0, imgs.shape[1], size=imgs.shape[0])
      imgs = imgs[range(imgs.shape[0]), rid, Ellipsis]
    # self.queue_all_images[:] = imgs
    # self.queue_all_labels[:] = labels
    # Push to queue.
    self.queue_all_images[-imgs.shape[0]:] = imgs
    self.queue_all_labels[-labels.shape[0]:] = labels
    self.queue_all_ids[-ids.shape[0]:] = ids

  def after_step(self, hooks):
    """Updates batch scores."""
    # update scores of current batch in training
    self._update_scores(*hooks)

  def init(self, hooks, iter_epoch):
    """Pre-fills the dictionary before training."""
    self.iter_epoch = iter_epoch
    n_per_class = self.batch_size // self.nclass
    self.ds_iter = tf.data.make_initializable_iterator(self.dataset_ds)
    ds_next_sample = self.ds_iter.get_next()
    self.sess.run(self.ds_iter.initializer)
    counts = np.array([0] * self.nclass)
    prefetch_labels = []
    prefetch_images = []
    while np.any(counts < n_per_class):
      # Keep balanced
      try:
        b_imgs, b_labels, _, _ = self.sess.run(ds_next_sample)
      except tf.errors.OutOfRangeError:
        # Happen when some classes do not have enough data to fill.
        break
      prefetch_labels.append(b_labels)
      tmp_id, tmp_c = np.unique(
          np.concatenate(prefetch_labels), return_counts=True)
      counts[tmp_id] = tmp_c
      prefetch_images.append(b_imgs)

    prefetch_labels = np.concatenate(prefetch_labels)
    prefetch_images = np.concatenate(prefetch_images)
    ids = self._cls_bal_sampling(prefetch_labels, self.batch_size)
    self._enqueue(prefetch_images[ids], prefetch_labels[ids], ids)
    self.hooks = hooks if hooks is not None else []

  def get_images_labels(self, best_ids):
    images, labels, ids, golds = [], [], [], []
    ds_next_example = self.ds_iter.get_next()
    self.sess.run(self.ds_iter.initializer)
    for _ in range(int(np.ceil(self.dataset_size / self.batch_size))):
      # look through the dataset to get data with best_ids
      try:
        b_imgs, b_labels, b_ids, b_gold_labels = self.sess.run(ds_next_example)
      except tf.errors.OutOfRangeError:
        break
      keep_ids = np.intersect1d(b_ids, best_ids, return_indices=True)[1]

      if np.size(keep_ids) > 0:
        images.append(b_imgs[keep_ids])
        labels.append(b_labels[keep_ids])
        ids.append(b_ids[keep_ids])
        golds.append(b_gold_labels[keep_ids])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)
    golds = np.concatenate(golds)
    return images, labels, ids, golds

  def after_epoch(self, epoch):
    """Updates probe_images and probe_labels dictionaries."""
    # Get top-k ranking per class
    with self.sess.as_default():
        self.global_gold_labels = self.queue_all_golden_labels.eval()
    if not np.all(self.ds_labels >= 0):
      t = np.sum(self.ds_labels < 0)
      logging.warning('ds_labels has {} unfilled items'.format(t))
    if not np.all(self.global_gold_labels >= 0):
      t = np.sum(self.ds_labels < 0)
      logging.warning('ds_labels has {} unfilled items'.format(t))

    best_ids = self._get_top_indices(self.capacity//self.nclass, self.batch_size//self.nclass)
    images, labels, ids, golds = self.get_images_labels(best_ids)
    mean_hit = (golds == labels).sum() / len(labels)
    if images.shape[0] != self.batch_size:
      logging.warning('{} != {}'.format(images.shape[0], self.batch_size))

    # Put to queue
    self._enqueue(images, labels, ids)

    max_score = self.ds_rates.max()
    min_score = self.ds_rates.min()
    self.purity_log.load(mean_hit)
    self.capacity_log.load(self.capacity)
    if best_ids.shape[0] >= self.init_capacity:
      # To prevent best_ids has shorter size and makes error.
      self.queue_data_indices.load(best_ids[:self.init_capacity])
    ratio = np.mean(self.ds_labels >= 0)

    logging.info(
        'Queue info at epoch {}: '
        'purity:{:.2f} ratio:{:.1f} cap:{} score(max/min):{:.2f}/{:.2f}'.format(
            epoch, mean_hit, ratio * 100., self.capacity, max_score, min_score))

    constraint_index = np.concatenate((self._get_top_or_bottom_indices(self.capacity // (2 * self.nclass)), self._get_top_or_bottom_indices(-self.capacity // (2 * self.nclass))))
    # constraint_index = self._get_active_learning_constraint(epoch, self.capacity)
    if len(constraint_index) < self.capacity:
      constraint_index = np.arange(self.capacity)
    images, labels, e_ids, golds = self.get_images_labels(constraint_index)
    self.constraint_index = constraint_index
    e_ids = e_ids.tolist()
    look_back = [e_ids.index(i) for i in constraint_index]
    self.queue_all_extract_images = images[look_back]
    self.queue_all_extract_labels = labels[look_back]
    golds = golds[look_back]
    self.constraint_is_correct = (self.queue_all_extract_labels == golds)
    np.savez(os.path.join(FLAGS.checkpoint_path, f"after2-epoch{epoch}.npz"), idx=constraint_index,labels=self.queue_all_extract_labels, probe_idx=ids)
    # np.savez(os.path.join(FLAGS.checkpoint_path, f"afterdata-epoch{epoch}.npz"), queue_all_images=self.queue_all_images,queue_all_labels=self.queue_all_labels,queue_all_extract_images=self.queue_all_extract_images, queue_all_extract_labels=self.queue_all_extract_labels)
    self.sess.run(
      [tf.assign(self.queue_probe_images, self.plh_probe_images),
      tf.assign(self.queue_probe_labels, self.plh_probe_labels),
      tf.assign(self.extract_images, self.plh_extract_images),
      tf.assign(self.extract_labels, self.plh_extract_labels),
      ],
      feed_dict={
          self.plh_probe_images: self.queue_all_images,
          self.plh_probe_labels: self.queue_all_labels,
          self.plh_extract_images: self.queue_all_extract_images,
          self.plh_extract_labels: self.queue_all_extract_labels,
      })

class OPT(L2R):
  """Fast sample re-weighting."""

  def __init__(self, sess, strategy, dataset):
    super(OPT, self).__init__(sess, strategy, dataset)
    self.step_hooks = []
    self.summaries = []
    with tf.variable_scope('weight'):
      # Accumulates smoothed per class weight
      self.class_weight_smooth = tf.get_variable(
          'smooth',
          trainable=False,
          shape=(self.dataset.num_classes,),
          initializer=tf.constant_initializer([1.0 / self.batch_size] *
                                              self.dataset.num_classes),
          dtype=tf.float32)

    if FLAGS.moving_weight_gamma:
      self.meta_weight_average = tf.get_variable(
          'moving_meta_weight',
          trainable=False,
          shape=(self.dataset.num_classes,),
          initializer=tf.constant_initializer([1.0 / self.batch_size] *
                                              self.dataset.num_classes),
          dtype=tf.float32)

  def callback_before_training(self, iter_epoch):
    # adding new returns for queue here
    self.queue.init(
        hooks=[self.labels, self.image_ids, self.logits, self.loss_margin],
        iter_epoch=iter_epoch)
    self.step_hooks.extend(self.queue.hooks)

  def callback_after_epoch(self, epoch):
    self.queue.after_epoch(epoch)
    if epoch < FLAGS.opt_start_epoch or epoch < 48 or 62 < epoch < 96 or 126 < epoch < 192:
      # if FLAGS.clean_opt:
      #   self.queue.lam = np.zeros(self.queue.batch_size)
      #   for l in range(self.queue.nclass):
      #     idx = (self.queue.queue_all_labels == l)
      #     self.queue.lam[idx] = 1 / self.queue.nclass / idx.sum()
      # else:
      self.queue.lam = np.ones(self.queue.batch_size) / self.queue.batch_size
      self.sess.run([tf.assign(self.queue.queue_probe_lam, self.queue.plh_probe_lam)],
        feed_dict={self.queue.plh_probe_lam: self.queue.lam})
    else:
      R = np.zeros((self.queue.batch_size, len(self.queue.queue_all_extract_images)))
      for i in range(self.queue.batch_size):
        R[i] = self.sess.run(self.extract_op, feed_dict={
          self.queue.plh_single_probe_images: self.queue.queue_all_images[[i]],
          self.queue.plh_single_probe_labels: self.queue.queue_all_labels[[i]],
        })[0]
      self.queue.lam = self.queue.get_lam(-R)
      self.sess.run([tf.assign(self.queue.queue_probe_lam, self.queue.plh_probe_lam)],
        feed_dict={self.queue.plh_probe_lam: self.queue.lam})
      np.savez(os.path.join(FLAGS.checkpoint_path, f"opt-epoch{epoch}.npz"), R=R, lam=self.queue.lam)

  def callback_after_step(self, step, hooks):
    self.queue.after_step(hooks)

  def callback_before_graph(self):
    self.queue = Queue(
        sess=self.sess,
        dataset_ds=self.dataset.train_dataflow,
        dataset_size=self.dataset.train_dataset_size,
        nclass=self.dataset.num_classes,
        capacity=FLAGS.queue_capacity,
        beta=FLAGS.queue_beta,
        batch_size=FLAGS.queue_bs,
        metric=FLAGS.queue_metric,
        shape=(self.dataset.image_size, self.dataset.image_size, 3))

  def compute_pseudo_loss(self, logits, img_ids, batch_weight):
    with tf.device('/cpu:0'):
      p_type, temp, gm, loss_w = FLAGS.use_pseudo_loss.split('_')
      temp, gm, loss_w = float(temp), float(gm), float(loss_w)
      logging.info('Pseudo loss info: \n\t type:{}  temp:{}  gamma:{}'.format(
          p_type, temp, gm))
      ema_logit = tf.gather(self.queue.queue_ema_logits, img_ids)
      ema_logit = logit_norm(ema_logit)
      # temp makes it sharpen?
      pseudo_labels = tf.nn.softmax(ema_logit / temp, name='pseudo_labels')
      p_net_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
          tf.stop_gradient(pseudo_labels), logits)
      if p_type == 'all':
        p_net_loss = tf.reduce_mean(p_net_cost, name='pseudo_loss')
      elif p_type == 'partial':
        # TODO(zizhaoz): remove not good!
        # Only supervised zero weight.
        # Deprecated. Not good at all.
        p_net_cost = tf.boolean_mask(p_net_cost, tf.equal(batch_weight, 0))
        p_net_loss = tf.reduce_sum(p_net_cost, name='pseudo_loss')
      else:
        raise NotImplementedError

      # Update ema_logits
      qqel = self.queue.queue_ema_logits
      updated_qqel = tf.scatter_update(qqel, img_ids,
                                       gm * ema_logit + (1 - gm) * logits)
      ops = [qqel.assign(updated_qqel)]

    return p_net_loss * loss_w, ops

  def set_input(self):
    train_ds = self.dataset.train_dataflow.shuffle(
        buffer_size=4096).repeat().batch(
            self.batch_size, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
    # TODO(zizhaoz): Adapts to all device types?
    # Ignore errors which happen rerely
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    self.train_input_iterator = self.strategy.make_dataset_iterator(train_ds)
    # self.train_input_iterator = self.strategy.experimental_distribute_dataset(
    #         joint_ds).make_initializable_iterator()
    val_ds = self.dataset.val_dataflow.batch(
        FLAGS.val_batch_size,
        drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    self.eval_input_iterator = self.strategy.make_dataset_iterator(val_ds)
    # self.eval_input_iterator = self.strategy.experimental_distribute_dataset(
    #         val_ds).make_initializable_iterator()
    if hasattr(self.dataset, 'test_dataflow'):
      self.enable_test_eval = True
      test_ds = self.dataset.test_dataflow.repeat().batch(
          FLAGS.val_batch_size, drop_remainder=True).prefetch(
              buffer_size=tf.data.experimental.AUTOTUNE)
      test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
      self.test_input_iterator = self.strategy.make_dataset_iterator(test_ds)

  def get_lookhead_variables(self, variables):
    """Extract intersting variables to lookahead."""
    all_vars = self.net.get_partial_variables(level=FLAGS.meta_partial_level)
    tf.logging.info('Get {} lookahead variables from {} totally \n {}'.format(
        len(all_vars), len(variables), all_vars))

    return all_vars

  def meta_momentum_update(self, grad, var_name, optimizer):
    """Momentum update."""
    accumulation = utils.get_var(optimizer.variables(), var_name.split(':')[0])
    if len(accumulation) != 1:
      raise ValueError('length of accumulation {}'.format(len(accumulation)))
    new_grad = tf.math.add(
        tf.stop_gradient(accumulation[0]) * FLAGS.meta_momentum, grad)
    return new_grad

  def entropy_minimization(self, logits):
    # entroy = - \sum_i^N 1/N * tf.math.log(p_i)
    p = tf.nn.softmax(logits)
    avg_p = tf.reduce_mean(p, 0)
    flat_labels = tf.ones_like(avg_p) / tf.cast(tf.shape(avg_p)[0], avg_p.dtype)
    entroy = tf.keras.losses.categorical_crossentropy(
        tf.expand_dims(flat_labels, 0), tf.expand_dims(avg_p, 0))
    return tf.reduce_sum(entroy)

  def meta_optimize(self, net_cost):
    """Meta optimization step."""
    probe_images, probe_labels, probe_lam = (self.queue.queue_probe_images,
                                  self.queue.queue_probe_labels,
                                  self.queue.queue_probe_lam)
    net = self.net
    gate_gradients = 1

    batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)
    if FLAGS.meta_momentum is None:
      # initial data weight is zero is default method
      init_eps_val = 0.0
    else:
      init_eps_val = 1.0 / batch_size

    meta_net = networks.MetaImage(self.net, name='meta_model')
    self.meta_net = meta_net

    if FLAGS.meta_momentum and not self.optimizer.variables():
      tmp_var_grads = self.optimizer.compute_gradients(
          tf.reduce_mean(net_cost), net.trainable_variables)
      self.optimizer.apply_gradients(tmp_var_grads)

    target = tf.constant(
        [init_eps_val] * batch_size, dtype=np.float32, name='weight')

    lookahead_loss = tf.reduce_sum(tf.multiply(target, net_cost))
    lookahead_loss = lookahead_loss + net.regularization_loss

    with tf.control_dependencies([lookahead_loss]):
      train_vars = self.get_lookhead_variables(net.trainable_variables)
      var_grads = tf.gradients(
          lookahead_loss, train_vars, gate_gradients=gate_gradients)

      static_vars = []
      for i in range(len(train_vars)):
        if FLAGS.meta_momentum > 0:
          actual_grad = self.meta_momentum_update(var_grads[i],
                                                  train_vars[i].name,
                                                  self.optimizer)
          static_vars.append(
              tf.math.subtract(
                  train_vars[i],
                  FLAGS.meta_stepsize * actual_grad,
                  name='meta/' + train_vars[i].name.rstrip(':0')))
        else:
          static_vars.append(
              tf.math.subtract(train_vars[i],
                               FLAGS.meta_stepsize * var_grads[i]))
        meta_net.add_variable_alias(
            static_vars[-1], var_name=train_vars[i].name)

      for uv in net.updates_variables:
        meta_net.add_variable_alias(
            uv, var_name=uv.name, var_type='updates_variables')
      meta_net.verbose()

    with tf.control_dependencies(static_vars):
      if 'margin' in FLAGS.queue_metric:
        # Forward self.images also to compute loss margin.
        all_logits = meta_net(
            tf.concat([probe_images, self.images], 0),
            name='meta_model',
            reuse=True,
            training=True)
        g_logits, train_logits = tf.split(
            all_logits, [tf.shape(probe_images)[0],
                         tf.shape(self.images)[0]])
        # Check the training images after graident descent for monitors
        meta_net_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            tf.one_hot(self.labels, self.dataset.num_classes), train_logits)
      else:
        g_logits = meta_net(
            probe_images, name='meta_model', reuse=True, training=True)
        meta_net_cost = tf.constant(0, tf.float32)


      desired_y = tf.one_hot(probe_labels, self.dataset.num_classes)
      if FLAGS.label_smoothing > 0:
        tf.logging.info('Use meta label smoothing %d' % FLAGS.label_smoothing)
        meta_cost = tf.losses.softmax_cross_entropy(
            desired_y,
            g_logits,
            label_smoothing=FLAGS.label_smoothing,
            reduction=tf.losses.Reduction.NONE)
      else:
        meta_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            desired_y, g_logits)
      #meta_loss = tf.reduce_mean(meta_cost, name='meta_loss')
      meta_loss = tf.reduce_sum(meta_cost * probe_lam)
      meta_loss = meta_loss + meta_net.get_regularization_loss(net.wd)
      meta_acc, meta_acc_op = tf.metrics.accuracy(probe_labels,
                                                  tf.argmax(g_logits, axis=1))

    with tf.control_dependencies([meta_loss] + [meta_acc_op]):
      meta_train_vars = meta_net.trainable_variables
      # Sanity: save memory for partial graph backpropagate
      grad_meta_vars = tf.gradients(
          meta_loss, meta_train_vars, gate_gradients=gate_gradients)
      grad_target = tf.gradients(
          static_vars,
          target,
          grad_ys=grad_meta_vars,
          gate_gradients=gate_gradients)[0]

    raw_weight = target - grad_target
    if FLAGS.clip_meta_weight:
      # Often used for noisy labels.
      raw_weight = raw_weight - init_eps_val
      unorm_weight = tf.clip_by_value(
          raw_weight, clip_value_min=0, clip_value_max=float('inf'))
    else:
      # Add init_eps_val to allow all data has non-zero weights
      unorm_weight = raw_weight - tf.reduce_min(raw_weight) + init_eps_val
    norm_c = tf.reduce_sum(unorm_weight)
    weight = tf.divide(unorm_weight, norm_c + 0.00001)

    if FLAGS.moving_weight_gamma:
      # Use moving average weight
      def get_updated_weight():
        moving_weight = tf.gather(self.meta_weight_average, self.labels)
        updates = FLAGS.moving_weight_gamma * moving_weight + (
            1 - FLAGS.moving_weight_gamma) * weight
        assign_op = self.meta_weight_average.scatter_nd_update(
            tf.expand_dims(self.labels, 1), updates)
        with tf.control_dependencies([assign_op]):
          updated_moving_weight = tf.gather(self.meta_weight_average,
                                            self.labels)
          return updated_moving_weight

      def get_fixed_weight():
        return tf.gather(self.meta_weight_average, self.labels)

      if FLAGS.meta_moving_end_epoch:
        epoch = tf.cast(self.global_step / self.iter_epoch, tf.int32)
        updated_moving_weight = tf.cond(
            tf.less(epoch, FLAGS.meta_moving_end_epoch), get_updated_weight,
            get_fixed_weight)
      else:
        updated_moving_weight = get_updated_weight()

      return tf.stop_gradient(updated_moving_weight
                             ), meta_cost, meta_acc, grad_target, meta_net_cost
    else:
      # Use raw weights.
      return tf.stop_gradient(
          weight), meta_cost, meta_acc, grad_target, meta_net_cost

  def extract_step(self):

    def extract_step_fn():
      """Step function."""
      images, labels = (self.queue.extract_images, self.queue.extract_labels)
      net = self.net
      logits = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      net_cost = tf.losses.softmax_cross_entropy(
          tf.one_hot(labels, self.dataset.num_classes),
          logits,
          label_smoothing=FLAGS.label_smoothing,
          reduction=tf.losses.Reduction.NONE)
      grad_target = self.meta_extract(net_cost)
      return tf.identity(grad_target)

    # End of parallel
    grad_target = self.strategy.run(extract_step_fn, args=())

    grad_target = self.strategy.unwrap(grad_target)
    grad_target = tf.concat(grad_target, axis=0)
    
    return [grad_target]
  def meta_extract(self, net_cost):
      net = self.net
      probe_images, probe_labels = (self.queue.plh_single_probe_images, self.queue.plh_single_probe_labels)
      gate_gradients = 1

      batch_size = FLAGS.queue_capacity
      init_eps_val = 1.0 / batch_size


      self.extract_net = networks.MetaImage(self.net, name='meta_model')
      meta_net = self.extract_net

      target = tf.constant(
          [init_eps_val] * batch_size, dtype=np.float32, name='extract_weight')

      lookahead_loss = tf.reduce_sum(tf.multiply(target, net_cost))
      lookahead_loss = lookahead_loss + net.regularization_loss

      with tf.control_dependencies([lookahead_loss]):
        train_vars = self.get_lookhead_variables(net.trainable_variables)
        var_grads = tf.gradients(
            lookahead_loss, train_vars, gate_gradients=gate_gradients)
        static_vars = []
        for i in range(len(train_vars)):
          static_vars.append(
              tf.math.subtract(train_vars[i],
                              FLAGS.meta_stepsize * var_grads[i]))
          meta_net.add_variable_alias(
              static_vars[-1], var_name=train_vars[i].name)

        for uv in net.updates_variables:
          meta_net.add_variable_alias(
              uv, var_name=uv.name, var_type='updates_variables')
        meta_net.verbose()

      with tf.control_dependencies(static_vars):
        all_logits = meta_net(
            tf.concat([probe_images, self.queue.extract_images], 0),
            name='meta_model',
            reuse=True,
            training=True)
        g_logits, train_logits = tf.split(
            all_logits, [tf.shape(probe_images)[0],
                          tf.shape(self.queue.extract_images)[0]])

        desired_y = tf.one_hot(probe_labels, self.dataset.num_classes)
        meta_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            desired_y, g_logits)
        meta_loss = tf.reduce_mean(meta_cost, name='meta_loss')
        meta_loss = meta_loss + meta_net.get_regularization_loss(net.wd)

      with tf.control_dependencies([meta_loss]):
        meta_train_vars = meta_net.trainable_variables
        grad_meta_vars = tf.gradients(
            meta_loss, meta_train_vars, gate_gradients=gate_gradients)
        grad_target = tf.gradients(
            static_vars,
            target,
            grad_ys=grad_meta_vars,
            gate_gradients=gate_gradients)[0]

      return grad_target



  def train_step(self):

    def step_fn(inputs):
      """Step function."""
      post_ops = []
      net = self.net
      if len(inputs) == 2:
        images, labels = inputs
        gold_labels = labels
      else:
        images, labels, self.image_ids, gold_labels = inputs

      # Some setting to help herited class to use
      if len(images.shape) == 5:
        # When strong augmentation is used.
        images, self.aug_images = images[:, 0], images[:, 1]
      else:
        self.aug_images = images

      # gold_labels only used for monitor only
      self.images, self.labels, self.gold_labels = images, labels, gold_labels

      if FLAGS.use_mixup:
        self.augment = MixMode()
        oh_labels = tf.one_hot(labels, self.dataset.num_classes)
        images, oh_labels = self.augment([images], [oh_labels], [0.5, 0.5])
      logits = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      self.logits = tf.identity(logits, name='logits')

      if FLAGS.use_mixup:
        net_cost = tf.losses.softmax_cross_entropy(
            oh_labels, logits, reduction=tf.losses.Reduction.NONE)
      else:
        net_cost = tf.losses.softmax_cross_entropy(
            tf.one_hot(labels, self.dataset.num_classes),
            logits,
            label_smoothing=FLAGS.label_smoothing,
            reduction=tf.losses.Reduction.NONE)

      if FLAGS.use_ema:
        ema_op = self.ema.apply(net.trainable_variables)

      (weight, meta_loss, meta_acc, grad_weight,
       meta_net_cost) = self.meta_optimize(net_cost)
      self.loss_margin = net_cost - meta_net_cost  # loss_margin_v2

      epoch = tf.cast(self.global_step / self.iter_epoch, tf.int32)
      batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)
      # When epoch < FLAGS.meta_start_epoch, we still compute meta_optimization
      weight = tf.cond(
          tf.greater_equal(epoch, FLAGS.meta_start_epoch), lambda: weight,
          lambda: tf.constant(  # pylint: disable=g-long-lambda
              [1.0 / batch_size] * batch_size, dtype=np.float32),
          'final_weight')
      net_loss = tf.reduce_sum(tf.math.multiply(net_cost, weight))
      qqaw = self.queue.queue_all_weights
      updated_qqaw = tf.scatter_update(qqaw, self.image_ids, weight)
      post_ops.extend([qqaw.assign(updated_qqaw)])
      qqasl = self.queue.queue_all_save_labels
      updated_qqasl = tf.scatter_update(qqasl, self.image_ids, self.labels)
      post_ops.extend([qqasl.assign(updated_qqasl)])
      qqagl = self.queue.queue_all_golden_labels
      updated_qqagl = tf.scatter_update(qqagl, self.image_ids, self.gold_labels)
      post_ops.extend([qqagl.assign(updated_qqagl)])

      if True: # more meta data
        qqal = self.queue.queue_all_loss
        updated_qqal = tf.scatter_update(qqal, self.image_ids, net_cost)
        post_ops.extend([qqal.assign(updated_qqal)])
        qqaml = self.queue.queue_all_meta_loss
        updated_qqaml = tf.scatter_update(qqaml, self.image_ids, meta_net_cost)
        post_ops.extend([qqaml.assign(updated_qqaml)])
        qqalm = self.queue.queue_all_loss_margin
        updated_qqalm = tf.scatter_update(qqalm, self.image_ids, self.loss_margin)
        post_ops.extend([qqalm.assign(updated_qqalm)])

      if FLAGS.use_pseudo_loss:
        p_loss, p_ops = self.compute_pseudo_loss(logits, self.image_ids, weight)
        p_loss = tf.cond(
            tf.greater_equal(epoch, FLAGS.meta_start_epoch), lambda: p_loss,
            lambda: tf.constant(0, dtype=np.float32))
        self.summaries.append(tf.summary.scalar('loss/pseudo', p_loss))
        net_loss += p_loss
        post_ops.extend(p_ops)

      net_loss += net.regularization_loss
      net_loss /= self.strategy.num_replicas_in_sync
      # Rescale by gpus
      net_grads = tf.gradients(net_loss, net.trainable_variables)
      minimizer_op = self.optimizer.apply_gradients(
          zip(net_grads, net.trainable_variables), global_step=self.global_step)
      if FLAGS.use_ema:
        optimizer_op = tf.group([net.updates, minimizer_op, ema_op])
      else:
        optimizer_op = tf.group([net.updates, minimizer_op])
      acc_op, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))
      with tf.control_dependencies([optimizer_op, acc_update_op] + post_ops):
        # Have to use tf.identity(). Why?!
        return (tf.identity(net_loss), tf.identity(meta_loss),
                tf.identity(meta_acc), tf.identity(acc_op), tf.identity(weight),
                tf.identity(labels), tf.identity(gold_labels),
                tf.identity(grad_weight), tf.identity(net_cost),
                tf.identity(meta_net_cost))

    # End of parallel
    (pr_net_loss, pr_metaloss, pr_metaacc, pr_acc, pr_weight, pr_labels,
     gold_labels, grad_weight, net_cost, meta_net_cost) = self.strategy.run(
         step_fn, args=(next(self.train_input_iterator),))

    # Collect device variables
    weights = self.strategy.unwrap(pr_weight)
    weights = tf.concat(weights, axis=0)
    labels = self.strategy.unwrap(pr_labels)
    labels = tf.concat(labels, axis=0)
    gold_labels = self.strategy.unwrap(gold_labels)
    gold_labels = tf.concat(gold_labels, axis=0)
    grad_weight = self.strategy.unwrap(grad_weight)
    grad_weight = tf.concat(grad_weight, axis=0)
    net_cost = self.strategy.unwrap(net_cost)
    net_cost = tf.concat(net_cost, axis=0)

    meta_net_cost = self.strategy.unwrap(meta_net_cost)
    meta_net_cost = tf.concat(meta_net_cost, axis=0)

    meta_cost = self.strategy.unwrap(pr_metaloss)
    meta_cost = tf.concat(meta_cost, axis=0)
    meta_loss = tf.reduce_mean(meta_cost)

    mean_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    mean_metaacc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaacc)
    net_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_net_loss)

    merges = []
    merges.append(tf.summary.scalar('acc/train', mean_acc))
    merges.append(tf.summary.scalar('loss/net', net_loss))
    merges.append(tf.summary.scalar('loss/meta', meta_loss))
    merges.append(tf.summary.scalar('acc/meta', mean_metaacc))

    zw_inds = tf.squeeze(
        tf.where(tf.less_equal(weights, 0), name='zero_weight_index'))
    merges.append(
        tf.summary.scalar(
            'weights/zeroratio',
            tf.math.divide(
                tf.cast(tf.size(zw_inds), tf.float32),
                tf.cast(tf.size(weights), tf.float32))))

    pmask = tf.equal(labels, gold_labels)
    nmask = tf.not_equal(labels, gold_labels)
    wmask = tf.not_equal(weights, 0.)
    weight_clean = reduce_mean(tf.boolean_mask(weights, pmask))
    weight_noisy = reduce_mean(tf.boolean_mask(weights, nmask))

    zeroweight_clean = reduce_mean(
        tf.boolean_mask(net_cost, tf.logical_and(pmask, tf.logical_not(wmask))))
    nonzeroweight_noisy = reduce_mean(
        tf.boolean_mask(net_cost, tf.logical_and(nmask, wmask)))
    merges.append(tf.summary.scalar('loss/masked_clean', zeroweight_clean))
    merges.append(tf.summary.scalar('loss/unmasked_noisy', nonzeroweight_noisy))
    merges.append(
        tf.summary.scalar('loss/clean',
                          reduce_mean(tf.boolean_mask(net_cost, pmask))))
    merges.append(
        tf.summary.scalar('loss/noisy',
                          reduce_mean(tf.boolean_mask(net_cost, nmask))))

    # loss margin information
    loss_margin = net_cost - meta_net_cost  # loss_margin_v2
    weight_smooth_op = []
    if FLAGS.verbose_finer_log:
      merges.append(
          tf.summary.scalar('loss_margin/clean',
                            reduce_mean(tf.boolean_mask(loss_margin, pmask))))
      merges.append(
          tf.summary.scalar('loss_margin/noisy',
                            reduce_mean(tf.boolean_mask(loss_margin, nmask))))

      merges.append(tf.summary.histogram('weights/value', weights))
      merges.append(tf.summary.scalar('weights/clean', weight_clean))
      merges.append(tf.summary.scalar('weights/noisy', weight_noisy))
      merges.append(
          tf.summary.scalar('weights/ratio',
                            reduce_mean(tf.cast(pmask, tf.float32))))
      passed = tf.reduce_sum(tf.cast(wmask, tf.float32))
      hit = tf.reduce_sum(tf.cast(tf.logical_and(pmask, wmask), tf.float32))
      merges.append(tf.summary.scalar('weights/purity', hit / passed))
      merges.append(tf.summary.histogram('weights/grad', grad_weight))

      # For imbalance experiments, enable class weight vis.
      nclass = self.dataset.num_classes
      for i in range(nclass):
        labels = tf.reshape(labels, shape=weights.shape)
        # pylint: disable=cell-var-from-loop
        avg_w = tf.cond(
            tf.reduce_any(tf.equal(labels, i)),
            # pylint: disable=g-long-lambda
            lambda: tf.reduce_mean(
                tf.boolean_mask(weights, tf.equal(labels, i)), axis=0),
            lambda: self.class_weight_smooth[i])
        weight_smooth_op.append(
            self.class_weight_smooth[i].assign(self.class_weight_smooth[i] *
                                               0.9 + avg_w * 0.1))
        merges.append(
            tf.summary.scalar('weights/class' + str(i),
                              self.class_weight_smooth[i]))

    # Validation results.
    if hasattr(self, 'eval_acc_on_train'):
      merges.append(
          tf.summary.scalar('acc/eval_on_traintime', self.eval_acc_on_train[0]))
      merges.append(
          tf.summary.scalar('acc/eval_on_traintime_top5',
                            self.eval_acc_on_train[1]))
      merges.append(
          tf.summary.scalar('acc/num_eval', self.eval_acc_on_train[2]))

    self.epoch_var = tf.cast(self.global_step / self.iter_epoch, tf.float32)
    merges.append(tf.summary.scalar('epoch', self.epoch_var))
    merges.append(tf.summary.scalar('learningrate', self.learning_rate))
    merges.extend(self.queue.summaries)
    merges.extend(self.summaries)
    summary = tf.summary.merge(merges)
    return [
        net_loss, meta_loss, mean_acc, mean_metaacc, summary, weights, grad_weight,
        weight_smooth_op
    ]


  def save_model(self, iteration):
    """Saves model."""
    self.queue.save(FLAGS.checkpoint_path, iteration)
    path = '{}/checkpoint.ckpt'.format(FLAGS.checkpoint_path)
    epoch = iteration // self.iter_epoch
    if epoch in [31,32,63,64,127,128,255,256,511,512]:
      save_path = self.saver.save(self.sess, path, global_step=iteration)
      logging.info('Save model weights {} at iteration {}'.format(
          save_path, iteration))

  def load_model(self, path=None):
    """Load model from disk if there is any or required by FLAGS.restore_step.

    Args:
      path: The path of checkpoints. If not provided, it will infer
        automatically by FLAGS.restore_step.
    """
    if path is None:
      path = self.check_checkpoint()
    if path is not None:
      self.saver.restore(self.sess, save_path=path)
      logging.info(
          'Load model checkpoint {}, learning_rate {:3f} global_step {}'.format(
              path, self.learning_rate.eval(), self.global_step.eval()))
      self.queue.load(FLAGS.checkpoint_path, self.global_step.eval())
    else:
      if FLAGS.mode == 'evaluation' and not FLAGS.pretrained_ckpt:
        raise ValueError('Checkpoint not found for evaluation')

  def train(self):
    self.set_input()
    self.callback_before_graph()
    self.build_graph()
    self.extract_op = self.extract_step()
    iter_epoch = self.iter_epoch
    with self.strategy.scope():
      self.initialize_variables()
      self.sess.run([self.train_input_iterator.initializer])
      self.sess.run([self.eval_input_iterator.initializer])
      if self.enable_test_eval:
        self.sess.run([self.test_input_iterator.initializer])
      self.saver = tf.train.Saver(max_to_keep=300)

      self.load_model()
      restore_step = self.global_step.eval()
      self.callback_before_training(iter_epoch)
      for iteration in range(restore_step + 1, FLAGS.max_iteration + 1):
        self.update_learning_rate(iteration)
        returns = self.sess.run([self.learning_rate] + self.train_op +
                                self.step_hooks)

        (lr, _, _, _, _, merged_summary, _, _, _) = returns[:-len(self.step_hooks)]
        self.callback_after_step(iteration, returns[-len(self.step_hooks):])
        if iteration % 10 == 0 or iteration == 1:
          # Reduce write frequency to reduce size of tf events.
          self.write_to_summary(self.summary_writer, merged_summary, iteration)

        if iteration % iter_epoch == 0:
          self.callback_after_epoch(iteration // iter_epoch)

        if self.time_for_evaluation(iteration, lr):
          self.evaluate(iteration, lr)
          if self.enable_test_eval:
            self.evaluate(iteration, lr, op=self.test_op, op_scope='test')
          self.save_model(iteration)
          self.write_to_summary(
              self.summary_writer, merged_summary, iteration, flush=True)
          gc.collect()

