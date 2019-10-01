import abc
from enum import Enum
import os
import tensorflow as tf
from .flowlib import flow_to_image, write_flow
import numpy as np
from scipy.misc import imread, imsave
import uuid
from .training_schedules import LONG_SCHEDULE
slim = tf.contrib.slim
import socket

class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.global_step = slim.get_or_create_global_step()
        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule, trainable=True):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    def init(self,checkpoint):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
    #######################
    # def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=False):
    #     input_a = imread(input_a_path)
    #     input_b = imread(input_b_path)
    #
    #     # Convert from RGB -> BGR
    #     input_a = input_a[..., [2, 1, 0]]
    #     input_b = input_b[..., [2, 1, 0]]
    #
    #     # Scale from [0, 255] -> [0.0, 1.0] if needed
    #     if input_a.max() > 1.0:
    #         input_a = input_a / 255.0
    #     if input_b.max() > 1.0:
    #         input_b = input_b / 255.0
    #
    #     # TODO: This is a hack, we should get rid of this
    #     training_schedule = LONG_SCHEDULE
    #
    #     inputs = {
    #         'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
    #         'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
    #     }
    #     predictions = self.model(inputs, training_schedule)
    #     pred_flow = predictions['flow']
    #
    #     saver = tf.train.Saver()
    #
    #     with tf.Session() as sess:
    #         saver.restore(sess, checkpoint)
    #         pred_flow = sess.run(pred_flow)[0, :, :, :]
    #
    #         unique_name = 'output_img'
    #         if save_image:
    #             flow_img = flow_to_image(pred_flow)
    #             full_out_path = os.path.join(out_path, unique_name + '.png')
    #             imsave(full_out_path, flow_img)
    #
    #         if save_flo:
    #             full_out_path = os.path.join(out_path, unique_name + '.flo')
    #             write_flow(pred_flow, full_out_path)
    ###################################
    def test(self, checkpoint, out_path, save_image=True, save_flo=True):
        #input_a = imread(input_a_path)
        #input_b = imread(input_b_path)

        # Convert from RGB -> BGR
        #input_a = input_a[..., [2, 1, 0]]
        #input_b = input_b[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        #if input_a.max() > 1.0:
        #    input_a = input_a / 255.0
        #if input_b.max() > 1.0:
        #    input_b = input_b / 255.0

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE
        input_a = tf.placeholder(shape = [384, 512, 3], dtype = tf.float32, name = "source_image")
        input_b = tf.placeholder(shape = [384, 512, 3], dtype = tf.float32, name = "target_image")
        inputs = {
            'input_a': tf.expand_dims(input_a, 0),
            'input_b': tf.expand_dims(input_b, 0),
        }
        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, checkpoint)
            HOST = '127.0.0.1'
            PORT = 50055
            #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #s.bind((HOST, PORT))
            #s.listen(1)
            #conn, addr = s.accept()


            while True:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((HOST, PORT))
                s.listen(1)
                conn, addr = s.accept()
                while True:
                    data = conn.recv(4096)
                    if not data: break
                    print()
                    conn.send(data)

                    if(str(data)=='0'):
                        source_path='/home/yvsharish/working/habitat-sim/image_baseline_2/test.rgba.00000.png'
                    else:
                        source_path = '/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.rgba.' + str(data).zfill(5) + '.png'
                    target_path = '/home/yvsharish/working/habitat-sim/image_baseline_2/test.rgba.00019.png'
                    source = imread(source_path)
                    target = imread(target_path)
                    source = source[..., [2, 1, 0]]
                    target = target[..., [2, 1, 0]]

                    # Scale from [0, 255] -> [0.0, 1.0] if needed
                    if source.max() > 1.0:
                        source = source / 255.0
                    if target.max() > 1.0:
                        target = target / 255.0
                    pred_flow_np = sess.run(pred_flow, feed_dict = {input_a : source,  input_b : target})[0, :, :, :]
                    save_flo = True
                    unique_name = 'test.flo.' + str(data).zfill(5)
                    actual_name='output_img'
                    if save_image:
                        flow_img = flow_to_image(pred_flow_np)
                        full_out_path = os.path.join('/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/', unique_name + '.png')
                        imsave(full_out_path, flow_img)

                    if save_flo:
                        full_out_path = os.path.join(out_path, actual_name + '.flo')
                        write_flow(pred_flow_np, full_out_path)
                conn.close()


    def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        if checkpoints:
            for (checkpoint_path, (scope, new_scope)) in checkpoints.iteritems():
                variables_to_restore = slim.get_variables(scope=scope)
                renamed_variables = {
                    var.op.name.split(new_scope + '/')[1]: var
                    for var in variables_to_restore
                }
                restorer = tf.train.Saver(renamed_variables)
                with tf.Session() as sess:
                    restorer.restore(sess, checkpoint_path)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            slim.learning.train(
                train_op,
                log_dir,
                # session_config=tf.ConfigProto(allow_soft_placement=True),
                global_step=self.global_step,
                save_summaries_secs=60,
                number_of_steps=training_schedule['max_iter']
            )
