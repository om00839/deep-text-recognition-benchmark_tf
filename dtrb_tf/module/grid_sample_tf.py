import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def grid_sample_2d(inp, grid):
    in_shape = tf.shape(inp)
    in_h = in_shape[1]
    in_w = in_shape[2]

    # Find interpolation sides
    i, j = grid[..., 0], grid[..., 1]
    i = tf.cast(in_h - 1, grid.dtype) * (i + 1) / 2
    j = tf.cast(in_w - 1, grid.dtype) * (j + 1) / 2
    i_1 = tf.maximum(tf.cast(tf.floor(i), tf.int32), 0)
    i_2 = tf.minimum(i_1 + 1, in_h - 1)
    j_1 = tf.maximum(tf.cast(tf.floor(j), tf.int32), 0)
    j_2 = tf.minimum(j_1 + 1, in_w - 1)

    # Gather pixel values
    n_idx = tf.tile(tf.range(in_shape[0])[:, tf.newaxis, tf.newaxis], tf.concat([[1], tf.shape(i)[1:]], axis=0))
    q_11 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_1], axis=-1))
    q_12 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_2], axis=-1))
    q_21 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_1], axis=-1))
    q_22 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_2], axis=-1))

    # Interpolation coefficients
    di = tf.cast(i, inp.dtype) - tf.cast(i_1, inp.dtype)
    di = tf.expand_dims(di, -1)
    dj = tf.cast(j, inp.dtype) - tf.cast(j_1, inp.dtype)
    dj = tf.expand_dims(dj, -1)

    # Compute interpolations
    q_i1 = q_11 * (1 - di) + q_21 * di
    q_i2 = q_12 * (1 - di) + q_22 * di
    q_ij = q_i1 * (1 - dj) + q_i2 * dj

    return q_ij

# Test it
inp = tf.placeholder(tf.float32, [None, None, None, None])
grid = tf.placeholder(tf.float32, [None, None, None, 2])
res = grid_sample_2d(inp, grid)
with tf.Session() as sess:
    # Make test image
    im_grid_i, im_grid_j = np.meshgrid(np.arange(6), np.arange(10), indexing='ij')
    im = im_grid_i + im_grid_j
    im = im / im.max()
    im = np.stack([im] * 3, axis=-1)
    # Test grid 1: complete image
    grid1 = np.stack(np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 18), indexing='ij'), axis=-1)
    # Test grid 2: lower right corner
    grid2 = np.stack(np.meshgrid(np.linspace(0, 1, 15), np.linspace(.5, 1, 18), indexing='ij'), axis=-1)
    # Run
    res1, res2 = sess.run(res, feed_dict={inp: [im, im], grid: [grid1, grid2]})
    # Plot image and sampled grids
    plt.figure()
    plt.imshow(im)
    plt.figure()
    plt.imshow(res1)
    plt.figure()
    plt.imshow(res2)