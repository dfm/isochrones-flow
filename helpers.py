# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["TFModel"]

import numpy as np

import tensorflow as tf


class TFModel(object):

    def __init__(self, target, var_list, feed_dict=None, session=None):
        self.target = target
        self.var_list = var_list
        self.grad_target = tf.gradients(self.target, self.var_list)
        self.feed_dict = {} if feed_dict is None else feed_dict
        self._session = session

    @property
    def session(self):
        if self._session is None:
            return tf.get_default_session()
        return self._session

    def value(self, vector):
        feed_dict = self.vector_to_feed_dict(vector)
        return self.session.run(self.target, feed_dict=feed_dict)

    def gradient(self, vector):
        feed_dict = self.vector_to_feed_dict(vector)
        return np.concatenate([
            np.reshape(g, s) for s, g in zip(
                self.sizes,
                self.session.run(self.grad_target, feed_dict=feed_dict))
        ])

    def setup(self, session=None):
        if session is not None:
            self._session = session
        values = self.session.run(self.var_list)
        self.sizes = [np.size(v) for v in values]
        self.shapes = [np.shape(v) for v in values]

    def vector_to_feed_dict(self, vector):
        i = 0
        fd = dict(self.feed_dict)
        for var, size, shape in zip(self.var_list, self.sizes, self.shapes):
            fd[var] = np.reshape(vector[i:i+size], shape)
            i += size
        return fd

    def feed_dict_to_vector(self, feed_dict):
        return np.concatenate([
            np.reshape(feed_dict[v], s)
            for v, s in zip(self.var_list, self.sizes)])

    def current_vector(self):
        values = self.session.run(self.var_list)
        return np.concatenate([
            np.reshape(v, s)
            for v, s in zip(values, self.sizes)])
