import argparse
import os
from ..net_new import Mode
from  .flownet2_new import FlowNet2
import socket
import tensorflow as tf

FLAGS = None

#net=FlowNet2(mode=Mode.TEST)
#checkpoint='/home/yvsharish/ICRA_2019/FlowNet2/flownet-2.ckpt-0'
#sess=tf.Session()
#with tf.Session(graph=tf.Graph()) as sess:
    #if(True):
        #net=FlowNet2(mode=Mode.TEST)


    #net.init(sess=sess,checkpoint='/home/yvsharish/ICRA_2019/FlowNet2/flownet-2.ckpt-0')
        #print('Init done')
    #HOST = '127.0.0.1'                 # Symbolic name meaning all available interfaces
    #PORT = 50007
def main():
    # Create a new network
    #net = FlowNet2(mode=Mode.TEST)
    with tf.Session() as sess:
    #if(True):
        net=FlowNet2(mode=Mode.TEST)
        print('Init done')
        HOST = '127.0.0.1'                 # Symbolic name meaning all available interfaces
        PORT = 50008
        cntr=1
        while True:
            print('In True outer')
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            s.listen(1)
            conn, addr = s.accept()
            print('Connected by', addr)
            while True:
                #net.init(sess=sess,checkpoint='/home/yvsharish/ICRA_2019/FlowNet2/flownet-2.ckpt-0')

                print('Inner True')
                data = conn.recv(4096)
            #data = data.split(" ")
                if not data: break
                print()
                net.test(
                        cntr,
                    sess=sess,
                    checkpoint='/home/yvsharish/ICRA_2019/FlowNet2/flownet-2.ckpt-0',
       # checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
                #input_a_path=FLAGS.input_a,
                #input_b_path=FLAGS.input_b,
                #out_path=FLAGS.out,
                    input_a_path='/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.rgba.'+str(data).zfill(5)+'.png',
                    input_b_path='/home/yvsharish/working/habitat-sim/image_baseline_2/test.rgba.00019.png',
                    out_path='/home/yvsharish/test/output_dir',

                    save_flo=True,
                )
                cntr=cntr+1

                conn.send(data)
            conn.close()
    # Train on the data
    #net.test(
        #checkpoint='/home/yvsharish/ICRA_2019/FlowNet2/flownet-2.ckpt-0',
       ## checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
        #input_a_path=FLAGS.input_a,
        #input_b_path=FLAGS.input_b,
        #out_path=FLAGS.out,
        #save_flo=True,
    #)


if __name__ == '__main__':
#    init()
    main()
