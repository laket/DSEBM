#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import time
import shutil

import tensorflow as tf

import config
from dataset.mnist import MNIST
from network import EBM
import metrics

import cv2

FLAGS = tf.app.flags.FLAGS


def eval():
    logger = logging.getLogger(__name__)

    dataset = MNIST(is_train=False, batch_size=FLAGS.batch_size)

    ### Network definition
    images, labels = dataset.dummy_inputs()
    ebm = EBM()
    energies = ebm.energy(images)

    #### Session setting
    save_dict = ebm.load_saver_dict()
    saver = tf.train.Saver(save_dict)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    list_energies = []
    list_images = []

    with tf.train.SingularMonitoredSession(config=session_config, checkpoint_dir=FLAGS.dir_parameter) as sess:
        num_iter = 0

        while not sess.should_stop():
            if dataset.completed:
                break

            cur_images, cur_labels = dataset.next_batch()

            cur_energies = sess.run(energies,
                                    feed_dict={images:cur_images})

            list_energies += cur_energies.tolist()
            list_images += cur_images.tolist()

    sorted_energies, sorted_images = zip(*sorted(zip(list_energies, list_images), reverse=True))

    count_image = 0
    for cur_energy, cur_image in zip(sorted_energies, sorted_images):
        count_image += 1
        cur_path = os.path.join(FLAGS.dir_eval, "{:04}.png".format(count_image))
        cur_image = dataset.depreprocess(cur_image)
        cv2.imwrite(cur_path, cur_image)


def main(argv=None):
    config.print_config()

    if os.path.exists(FLAGS.dir_eval):
        shutil.rmtree(FLAGS.dir_eval)
    os.makedirs(FLAGS.dir_eval)

    eval()

if __name__ == "__main__":
    tf.app.run()
