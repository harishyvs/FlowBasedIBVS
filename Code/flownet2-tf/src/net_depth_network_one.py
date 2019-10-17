import abc
from enum import Enum
import os
import tensorflow as tf
from .flowlib import flow_to_image, write_flow
import numpy as np
from scipy.misc import imread, imsave
import uuid
from PIL import Image, ImageFile

from .training_schedules import LONG_SCHEDULE
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import time
import threading
import socket
slim = tf.contrib.slim

def load_image(image_file):
    image = cv2.resize(cv2.imread(file), (640, 480), interpolation = cv2.INTER_AREA)
    x = np.clip(image/ 255, 0, 1)
    return np.expand_dims(x, axis=0)

def DepthNorm(x, maxDepth):
    return maxDepth / x

class Mode(Enum):
    TRAIN = 1
    TEST = 2

def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

    return np.stack(scaled)

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

    def test(self, checkpoint,out_path, save_image=True, save_flo=True):

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE
        input_a = tf.placeholder(shape = [1,384, 512, 3], dtype = tf.float32, name = "source_image")
        input_b = tf.placeholder(shape = [1,384, 512, 3], dtype = tf.float32, name = "target_image")
        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }

        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        # define two sessions

        sess_flow = tf.Session()
        #sess_ = tf.Session()

        # global initialisation

        init_op_f = tf.global_variables_initializer()
        sess_flow.run(init_op_f)

        #load the flow model

        saver.restore(sess_flow, checkpoint)

        # load the depth model

        f = gfile.FastGFile("./src/flownet2/FlowNet2/tf_model.pb", 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        f.close()
        sess_flow.graph.as_default()
        tf.import_graph_def(graph_def)

        # depth computation graph

        input_image = sess_flow.graph.get_tensor_by_name('import/input_1:0')
        pred_depth_tensor = sess_flow.graph.get_tensor_by_name('import/conv3/BiasAdd:0')
        HOST = '127.0.0.1'
        PORT = 50055
        
        # inference input preprocessing
        while True:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            s.listen(1)
            conn, addr = s.accept()
            flowfile=open("./src/aaaaa/baseline_2_avg_flow.txt","a+")
            while True:
                data = conn.recv(4096)
                if not data: break
                print()
                conn.send(data)
                if(str(data)=='0'):
                    input_a_path='./src/image_baseline_2/initial_image.png'
                else:
                    input_a_path = './src/image_baseline_2_output/test.rgba.' + str(data).zfill(5) + '.png'
                input_b_path = './src/image_baseline_2/desired_image.png'

                source =cv2.imread(input_a_path,cv2.COLOR_BGR2RGB)
                target =cv2.imread(input_b_path,cv2.COLOR_BGR2RGB)

                source = source / 255.0
                inp_source = np.expand_dims(np.clip(cv2.resize(source, (640, 480), interpolation = cv2.INTER_AREA), 0, 1), axis = 0)
                target = target / 255.0
                source = source[..., [2, 1, 0]]
                target = target[..., [2, 1, 0]]
                source = np.expand_dims(source, 0)
                target = np.expand_dims(target, 0)
                predicted_flow, predicted_depth = sess_flow.run((pred_flow, pred_depth_tensor), feed_dict = {input_a : source,  input_b : target, input_image : inp_source})
                pred_depth = np.clip(DepthNorm(predicted_depth, maxDepth=1000), 10, 1000) / 1000
                pred_depth = scale_up(2, pred_depth[:,:,:,0]) * 100.0

                """
                source = source[..., [2, 1, 0]]
                target = target[..., [2, 1, 0]]
                inp_source=np.clip(inp_source/255,0,1)
                if source.max() > 1.0:
                    source = source / 255.0
                if target.max() > 1.0:
                    target = target / 255.0
                predicted_flow = 0
                predicted_depth = 0
                feed_dict_flow = {input_a : source, input_b : target}
                feed_dict_depth = {input_image : inp_source}
                #predicted_flow = sess_flow.run(pred_flow, feed_dict = feed_dict_flow)
                predicted_flow, predicted_depth = sess_flow.run((pred_flow, pred_depth_tensor), feed_dict = {input_a : source, input_b : target, input_image : inp_source})


                pred_depth = np.clip(DepthNorm(predicted_depth, maxDepth=1000), 10, 1000) / 1000
                pred_depth = scale_up(2, pred_depth[:,:,:,0]) * 10.0
                """
                depth_img = np.zeros((480,640,1))
                depth_img[:, :, 0] = pred_depth[0, :, :]
                pred_flow_np = predicted_flow[0, :, :, :]
                depth_img=cv2.resize(pred_depth[0, :, :], (512, 384), interpolation = cv2.INTER_AREA)
                unique_name = 'test.flo.' + str(data).zfill(5)
                actual_name='output_img'
                if save_image:
                    flow_img, avg_flow = flow_to_image(pred_flow_np)
                    flowfile.write("Average flow - depth-network"+str(data).zfill(5)+":"+str(avg_flow))
                    full_out_path = os.path.join('./src/image_baseline_2_output/', unique_name + '.png')
                    imsave(full_out_path, flow_img)

                if save_flo:
                    full_out_path = os.path.join(out_path, actual_name + '.flo')
                    write_flow(pred_flow_np, full_out_path)

                shan='test.depth.'
                full_out_path = os.path.join('./src/image_baseline_2_output/',shan+str(data).zfill(5)+ '.png')
                imsave(full_out_path, depth_img)
                print('image saved at desired location  ')
                image = np.asarray(imread(full_out_path))
                print(np.shape(image))
            flowfile.close()
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
