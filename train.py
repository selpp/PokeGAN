import tensorflow as tf
import network as net
import numpy as np
import os

from preprocess import save
from tqdm import tqdm

def train(dir, random_dim, width, height, channels, batch_size, epoch):
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape = [None, height, width, channels], name = 'real_image')
        random_input = tf.placeholder(tf.float32, shape = [None, random_dim], name = 'random_input')

    fake_image = net.generator(random_input, channels, random_dim, is_train = True)
    real_result, _ = net.discriminator(real_image, is_train = True)
    fake_result, _ = net.discriminator(fake_image, is_train = True, reuse = True)

    fake_result_mean = tf.reduce_mean(fake_result)

    d_loss = tf.reduce_mean(real_result) - fake_result_mean
    g_loss = -fake_result_mean

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]

    learning_rate = 1e-3

    trainer_d = tf.train.AdamOptimizer(learning_rate).minimize(-d_loss, var_list = d_vars)
    trainer_g = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = g_vars)

    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    images_batch, samples_num = net.get_images_batch(dir, width, height, channels, batch_size)

    batch_num = int(samples_num / batch_size)
    total_batch = 0

    sess = tf.Session()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('logs/newPokemon/', sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    save_path = saver.save(sess, '/tmp/model.ckpt')
    ckpt = tf.train.latest_checkpoint('./model/newPokemon')
    saver.restore(sess, ckpt)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    tf.summary.scalar('loss_discriminator', d_loss)
    tf.summary.scalar('loss_generator', g_loss)
    summary_op = tf.summary.merge_all()

    print('total training sample num: %d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, epoch))
    print('start training...')

    for i in tqdm(range(epoch)):
        for j in range(batch_num):
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size = [batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                train_image = sess.run(images_batch)
                sess.run(d_clip)

                _, dLoss = sess.run([trainer_d, d_loss], feed_dict = {random_input: train_noise, real_image: train_image})

            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss], feed_dict = {random_input: train_noise})

        if i == 0:
            if not os.path.exists('./newPokemon'):
                os.makedirs('./newPokemon')
            for index in range(train_image.shape[0]):
                image = train_image[index]
                save(image, './newPokemon/batch' + str(i) + '_image' + str(index) + '.jpg')
        if i % 100 == 0:
            if not os.path.exists('./model/newPokemon'):
                os.makedirs('./model/newPokemon')
            saver.save(sess, './model/newPokemon/' + str(i))
        if i % 50 == 0:
            if not os.path.exists('./newPokemon'):
                os.makedirs('./newPokemon')
            sample_noise = np.random.uniform(-1.0, 1.0, size = [10, random_dim]).astype(np.float32) #[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict = {random_input: sample_noise})
            for index in range(imgtest.shape[0]):
                image = imgtest[index]
                save(image, './newPokemon/epoch' + str(i) + '_image' + str(index) + '.jpg')

            summary_str = sess.run(summary_op, feed_dict = {random_input: train_noise, real_image: train_image})
            writer.add_summary(summary_str, i)
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))

    coord.request_stop()
    coord.join(threads)
