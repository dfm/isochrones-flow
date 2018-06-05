#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from itertools import product
from scipy.interpolate import RegularGridInterpolator

import tensorflow as tf

session = tf.InteractiveSession()

if len(sys.argv) > 1:
    mist_path = sys.argv[1]
else:
    mist_path = os.path.expanduser("~/.isochrones/mist")

path = os.path.expanduser(os.path.join(
    mist_path, "MIST_v1.1_vvcrit0.0_UBVRIplus/*.iso.cmd"))
df = None
columns = None
for fn in glob.iglob(path):
    if columns is None:
        with open(fn, "r") as f:
            for line in f:
                if line.startswith("# EEP"):
                    columns = line[2:].split()
    assert columns is not None
    df0 = pd.read_table(fn, comment="#", delim_whitespace=True,
                        skip_blank_lines=True, names=columns)
    df0["feh"] = df0["[Fe/H]_init"]
    df0["age"] = df0["log10_isochrone_age_yr"]
    df0["eep"] = df0["EEP"]
    df0 = df0[df0.age > 8]
    if df is None:
        df = df0
    else:
        df = pd.concat((df, df0))

bands = ['Gaia_G_DR2Rev', 'Gaia_BP_DR2Rev', 'Gaia_RP_DR2Rev', '2MASS_J',
         '2MASS_H', '2MASS_Ks', 'Bessell_U', 'Bessell_V']

other_columns = ["star_mass", "initial_mass", "log_g", "log_Teff"]
all_columns = bands + other_columns
x_cols = ["feh", "age", "eep"]
points = []
inds = []
for c in x_cols:
    x = np.sort(np.array(df[c].unique())).astype(float)
    points.append(x)
    inds_x = np.searchsorted(x, np.array(df[c]))
    inds.append(inds_x)

values = np.empty([len(p) for p in points] + [len(all_columns)])
values[:] = np.nan
values[tuple(inds)] = np.array(df[all_columns])


middle = np.zeros([len(p) - 1 for p in points] + [len(all_columns)])
count = 0
axes = [np.arange(len(p) - 1) for p in points]
for inds in product(*([0, 1] for p in points)):
    select = [a + i for i, a in zip(inds, axes)]
    middle += values[np.meshgrid(*select, indexing="ij")]
    count += 1
middle /= count
log_vol = np.sum(np.meshgrid(*(np.log(np.diff(p)) for p in points),
                             indexing="ij"), axis=0)
middle = middle.reshape((-1, middle.shape[-1]))
log_vol = log_vol.flatten()
m = np.all(np.isfinite(middle), axis=1)
middle = np.ascontiguousarray(middle[m])
log_vol = np.ascontiguousarray(log_vol[m])

np.random.seed(42)
model = RegularGridInterpolator(points, values)
z_true = np.nan + np.zeros(len(all_columns))
while np.any(np.isnan(z_true)):
    z_true = model([np.random.uniform(p[0], p[-1]) for p in points])[0]

mag_obs = z_true[:len(bands)] + 5.0 * np.log10(10.0) - 5.0
mag_err = 0.01 * np.ones_like(mag_obs)
mag_obs += mag_err * np.random.randn(len(mag_obs))


T = tf.float64
log10_dist = tf.Variable(np.log10(10.0), dtype=T)
params = [log10_dist]

dm = 5.0 * log10_dist - 5.0
mag = middle[:, :len(bands)] + dm
chi2 = tf.reduce_sum(tf.square((mag - mag_obs[None, :]) / mag_err[None, :]),
                     axis=1)

log_like = tf.reduce_logsumexp(-0.5*chi2+log_vol)
grad_log_like = tf.gradients(log_like, params)
log_prob = log_like
opt = tf.contrib.opt.ScipyOptimizerInterface(-log_prob, params)
session.run(tf.global_variables_initializer())

print(session.run(log_like))
K = 10
strt = time.time()
for _ in range(K):
    session.run(log_like)
end = time.time()
print((end - strt) / K)


print(session.run(grad_log_like))
K = 10
strt = time.time()
for _ in range(K):
    session.run(grad_log_like)
end = time.time()
print((end - strt) / K)
