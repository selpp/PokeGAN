import tensorflow as tf
import network as net
import numpy as np
import os

from preprocess import save
from tqdm import tqdm

def test(random_dim, width, height, channels):
	batch_size = 32**2

	with tf.variable_scope('input'):
		random_input = tf.placeholder(tf.float32, shape = [None, random_dim], name = 'random_input')

	fake_image = net.generator(random_input, channels, random_dim, is_train = True)

	sess = tf.Session()
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter('logs/newPokemon/', sess.graph)

	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	save_path = saver.save(sess, '/tmp/model.ckpt')
	ckpt = tf.train.latest_checkpoint('./model/newPokemon')
	saver.restore(sess, ckpt)

	if not os.path.exists('./test'):
		os.makedirs('./test')
	sample_noise = np.random.uniform(-1.0, 1.0, size = [batch_size, random_dim]).astype(np.float32)
	imgtest = sess.run(fake_image, feed_dict = {random_input: sample_noise})

	final_img = np.vstack([np.hstack([imgtest[i * j] for j in range(32)]) for i in range(32)])

	save(final_img, './test/mozaic.jpg')
