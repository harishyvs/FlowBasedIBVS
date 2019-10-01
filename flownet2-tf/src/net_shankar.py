import abc
from enum import Enum
import os
import tensorflow as tf
from .flowlib import flow_to_image, write_flow
import numpy as np
from scipy.misc import imread, imsave
import uuid
from .training_schedules import LONG_SCHEDULE
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import time
import threading
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

    def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=False):

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE
        
        #define placeholders
        
        input_tensor_a = tf.placeholder(shape = [1, 384, 512, 3], dtype = tf.float32, name = "source_image")
        input_tensor_b = tf.placeholder(shape = [1, 384, 512, 3], dtype = tf.float32, name = "target_image")

        # flow computation graph

        input_tensors = {
            'input_a': input_tensor_a,
            'input_b': input_tensor_b,
        }
        
        predictions = self.model(input_tensors, training_schedule)
        pred_flow = predictions['flow']
        saver = tf.train.Saver()

        # define two sessions

        sess_flow = tf.Session()
        #sess_depth = tf.Session()
            
        # global initialisation

        init_op_f = tf.global_variables_initializer()
        sess_flow.run(init_op_f)
        
        #load the flow model

        saver.restore(sess_flow, checkpoint)   
        
        # load the depth model

        f = gfile.FastGFile("/home/nsaishankar/ICRA_2019_VS/DenseDepth/model/tf_model.pb", 'rb') 
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())
        f.close()
        sess_flow.graph.as_default()
        tf.import_graph_def(graph_def)    
            
        # depth computation graph
        
        input_image = sess_flow.graph.get_tensor_by_name('import/input_1:0')
        pred_depth_tensor = sess_flow.graph.get_tensor_by_name('import/conv3/BiasAdd:0')
            
        # inference input preprocessing
            
        source = imread(input_a_path)
        target = imread(input_b_path)
        source = source / 255.0
        inp_source = np.expand_dims(np.clip(cv2.resize(source, (640, 480), interpolation = cv2.INTER_AREA), 0, 1), axis = 0)
        target = target / 255.0
        source = source[..., [2, 1, 0]]
        target = target[..., [2, 1, 0]]
        source = np.expand_dims(source, 0)
        target = np.expand_dims(target, 0)
            
        # inference outputs
        """
        predicted_flow = 0
        predicted_depth = 0
        
        feed_dict_flow = {input_tensor_a : source, input_tensor_b : target}
        feed_dict_depth = {input_image : inp_source}

        # define compute flow

        def compute_flow():
            #print("1")
            #print(time.time())
            predicted_flow = sess_flow.run(pred_flow, feed_dict = feed_dict_flow)
            #print(np.shape(predicted_flow))
            #print("2")
            #print(time.time())
        # define compute depth

        def compute_depth():
            #print("3")
            #print(time.time())
            predicted_depth = sess_depth.run(pred_depth_tensor, feed_dict = feed_dict_depth)
            #print(np.shape(predicted_depth))
            #print("4")
            #print(time.time())
        # initialise two threads

        flow_thread = threading.Thread(target = compute_flow)
        depth_thread = threading.Thread(target = compute_depth)

        # start both the threads
    
        flow_thread.start()
        depth_thread.start()

        flow_thread.join()
        depth_thread.join()

        start = time.time()
        for i in range(100):
            print(i)
            flow_thread = threading.Thread(target = compute_flow)
            depth_thread = threading.Thread(target = compute_depth)
            flow_thread.start()
            #depth_thread.start()
            flow_thread.join()
            #depth_thread.join()

        end = time.time()
        print(end - start)
        """
        predicted_flow, predicted_depth = sess_flow.run((pred_flow, pred_depth_tensor), feed_dict = {input_tensor_a : source,  input_tensor_b : target, input_image : inp_source})
        """                        
        print('starting')
        start = time.time()
          
        for i in range(100):
            print(i)
            predicted_flow, predicted_depth = sess.run((pred_flow, pred_depth_tensor), feed_dict = {input_tensor_a : source, input_tensor_b : target, input_image : inp_source})
                   
        end = time.time() 
        print(str(end - start))
        """
        print(predicted_depth)
        pred_depth = np.clip(DepthNorm(predicted_depth, maxDepth=1000), 10, 1000) / 1000 
        pred_depth = scale_up(2, pred_depth[:,:,:,0]) * 10.0
        print(np.shape(pred_depth))
        depth_img = np.zeros((480, 640, 3))
        depth_img[:, :, 0] = pred_depth[0, :, :]
        depth_img[:, :, 1] = pred_depth[0, :, :]
        depth_img[:, :, 2] = pred_depth[0, :, :]
        print(depth_img)

        pred_flow_np = predicted_flow[0, :, :, :]
        unique_name = 'flow-' + str(uuid.uuid4())
            
        if save_image:
            flow_img = flow_to_image(pred_flow_np)
            full_out_path = os.path.join(out_path, unique_name + '.png')
            imsave(full_out_path, flow_img)

        if save_flo:
            full_out_path = os.path.join(out_path, unique_name + '.flo')
            write_flow(pred_flow, full_out_path)
            
        full_out_path = os.path.join(out_path, 'depth_' + '.png')
        imsave(full_out_path, depth_img)
        

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
