import tensorflow as tf

class SR4DFlowNet():
    def __init__(self, res_increase):
        self.res_increase = res_increase

    def build_network(self, u, v, w, u_mag, v_mag, w_mag, channel_nr=64, low_resblock=8, hi_resblock=4):
        # Prepare input layers
        speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
        mag = (u_mag ** 2 + v_mag ** 2 + w_mag ** 2) ** 0.5
        pcmr = mag * speed

        phase = tf.concat((u,v,w), axis=4)
        pc = tf.concat((pcmr, mag, speed), axis=4)

        # Magnitude input
        pc = self.conv3d(pc,3,1,channel_nr, 'SYMMETRIC', tf.nn.relu)
        pc = self.conv3d(pc,3,1,channel_nr, 'SYMMETRIC', tf.nn.relu)

        # Velocity input
        phase = self.conv3d(phase,3,1,channel_nr, 'SYMMETRIC', tf.nn.relu)
        phase = self.conv3d(phase,3,1,channel_nr, 'SYMMETRIC', tf.nn.relu)

        # Concat layer
        concat_layer = tf.concat((phase, pc), axis=4)
        concat_layer = self.conv3d(concat_layer, 1, 1, channel_nr, 'SYMMETRIC', tf.nn.relu)
        concat_layer = self.conv3d(concat_layer, 3, 1, channel_nr, 'SYMMETRIC', tf.nn.relu)
        
        # LR space Resblocks
        rb = concat_layer
        for i in range(low_resblock):
            rb = self.resnet_block(rb, "ResBlock", channel_nr, pad='SYMMETRIC')

        # Upsample
        rb = self.upsample3d(rb)
            
        # HR space Resblocks
        for i in range(hi_resblock):
            rb = self.resnet_block(rb, "ResBlock", channel_nr, pad='SYMMETRIC')

        # Split into 3 separate path, 1 path for each velocity component
        u_path = self.conv3d(rb, 3, 1, channel_nr, 'SYMMETRIC', tf.nn.relu)
        u_path = self.conv3d(u_path, 3, 1, 1, 'SYMMETRIC', tf.nn.tanh)

        v_path = self.conv3d(rb, 3, 1, channel_nr, 'SYMMETRIC', tf.nn.relu)
        v_path = self.conv3d(v_path, 3, 1, 1, 'SYMMETRIC', tf.nn.tanh)

        w_path = self.conv3d(rb, 3, 1, channel_nr, 'SYMMETRIC', tf.nn.relu)
        w_path = self.conv3d(w_path, 3, 1, 1, 'SYMMETRIC', tf.nn.tanh)

        # Combined them together
        b_out = tf.concat((u_path, v_path, w_path), axis=4)
        return b_out

    def upsample3d(self, input_tensor):
        """
            Resize the image by linearly interpolating the input
            using TF '``'resize_bilinear' function.

            :param input_tensor: 2D/3D image tensor, with shape:
                'batch, X, Y, Z, Channels'
            :return: interpolated volume

            Original source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
        """
        
        # We need this option for the bilinear resize to prevent shifting bug
        align = True 
        with tf.name_scope('UpsampleLayer'):
            b_size, x_size, y_size, z_size, c_size = input_tensor.shape.as_list()

            # Do this in tensor shape, except for channel size, otherwise we cannot use the input shape dynamically
            # Based on : https://stackoverflow.com/questions/55814061/resizing-images-with-dynamic-shape-in-tensorflow
            shape = tf.cast(tf.shape(input_tensor), dtype=tf.int32)
            b_size = tf.cast(shape[0], dtype=tf.int32)
            x_size = tf.cast(shape[1], dtype=tf.int32)
            y_size = tf.cast(shape[2], dtype=tf.int32)
            z_size = tf.cast(shape[3], dtype=tf.int32)
            # Channel = shape[4] size is known, so treat it as actual int, not as tensor

            x_size_new, y_size_new, z_size_new = x_size * self.res_increase, y_size * self.res_increase, z_size * self.res_increase

            if (x_size == x_size_new) and (y_size == y_size_new) and (z_size == z_size_new):
                # already in the target shape
                return input_tensor

            # resize y-z
            squeeze_b_x = tf.reshape(input_tensor, [-1, y_size, z_size, c_size])
            resize_b_x = tf.image.resize_bilinear(squeeze_b_x, [y_size_new, z_size_new], align_corners=align)            
            resume_b_x = tf.reshape(resize_b_x, [-1, x_size, y_size_new, z_size_new, c_size])

            # Reorient
            reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
            
            # Resize x
            squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size])
            resize_b_z = tf.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new], align_corners=align)
            resume_b_z = tf.reshape(resize_b_z, [-1, z_size_new, y_size_new, x_size_new, c_size])

            output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
            return output_tensor

    def conv3d(self, inputs, kernel_size, strides, filters, padding='SYMMETRIC', activation=None, use_bias=True):
        """
            Based on: https://github.com/gitlimlab/CycleGAN-Tensorflow/blob/master/ops.py
            For tf padding, refer to: https://www.tensorflow.org/api_docs/python/tf/pad

        """
        if padding == 'SYMMETRIC' or padding == 'REFLECT':
            p = (kernel_size - 1) // 2
            x = tf.pad(inputs, [[0,0],[p,p],[p,p], [p,p],[0,0]], padding)
            x = tf.layers.conv3d(inputs=x, kernel_size=kernel_size, strides=strides, filters=filters, padding='VALID', activation=activation, use_bias=use_bias)
        else:
            assert padding in ['SAME', 'VALID']
            x = tf.layers.conv3d(inputs=inputs, kernel_size=kernel_size, strides=strides, filters=filters, padding=padding, activation=activation, use_bias=use_bias)

        return x

    def resnet_block(self, x, block_name='ResBlock', channel_nr=64, scale = 1, pad='SAME'):
        tmp = self.conv3d(inputs=x, kernel_size=3, strides=1, filters=channel_nr, padding=pad, activation=None, use_bias=False)
        tmp = tf.nn.leaky_relu(tmp)

        tmp = self.conv3d(inputs=tmp, kernel_size=3, strides=1, filters=channel_nr, padding=pad, activation=None, use_bias=False)

        tmp = x + tmp * scale
        tmp = tf.nn.leaky_relu(tmp)
    
        return tmp
