import tensorflow as tf
import SR4DFlowNet as network

class SRLoader():
    def __init__(self, model_dir, patch_size, res_increase):
        tf.reset_default_graph()

        self.session = tf.Session()

        # Placeholders
        self.u = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="u")
        self.v = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="v")
        self.w = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="w")
        
        self.u_mag = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="u_mag")
        self.v_mag = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="v_mag")
        self.w_mag = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="w_mag")

        u = tf.expand_dims(self.u, 4)
        v = tf.expand_dims(self.v, 4)
        w = tf.expand_dims(self.w, 4)

        u_mag = tf.expand_dims(self.u_mag, 4)
        v_mag = tf.expand_dims(self.v_mag, 4)
        w_mag = tf.expand_dims(self.w_mag, 4)

        # network & output
        net = network.SR4DFlowNet(res_increase)
        self.predictions = net.build_network(u, v, w, u_mag, v_mag, w_mag)

        # print('Restoring 4DFlowNet')
        saver = tf.train.Saver()
        saver.restore(self.session, tf.train.latest_checkpoint(model_dir))

    def predict(self, u, v, w, u_mag, v_mag, w_mag):
        feed_dict = { self.u: u, self.v: v, self.w: w,
                      self.u_mag: u_mag, self.v_mag: v_mag, self.w_mag: w_mag }
        srImages = self.session.run(self.predictions, feed_dict)
        
        return srImages
