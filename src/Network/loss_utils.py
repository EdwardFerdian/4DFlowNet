import tensorflow as tf
import numpy as np

def create_divergence_kernels():
    """
        Create kernels in 3 different direction to calculate central differences
        The kernels will filter out the x, y, z direction vector
        Representing the gradients for each direction
    """
    kernel_x = np.zeros((3,3,3), dtype='float32')
    kernel_x[0,1,1] = 1 # filter x
    kernel_x[2,1,1] = -1 # filter x
    filter_x = tf.constant(kernel_x, dtype='float32')
    filter_x = tf.reshape(filter_x, [3, 3, 3, 1, 1])

    kernel_y = np.zeros((3,3,3), dtype='float32')
    kernel_y[1,0,1] = 1 # filter y
    kernel_y[1,2,1] = -1 # filter y
    filter_y = tf.constant(kernel_y, dtype='float32')
    filter_y = tf.reshape(filter_y, [3, 3, 3, 1, 1])

    kernel_z = np.zeros((3,3,3), dtype='float32')
    kernel_z[1,1,0] = 1 # filter z
    kernel_z[1,1,2] = -1 # filter z
    filter_z = tf.constant(kernel_z, dtype='float32')
    filter_z = tf.reshape(filter_z, [3, 3, 3, 1, 1])

    return (filter_x, filter_y, filter_z)    

def calculate_gradient(image, kernel):
    """
        Calculate the gradient (edge) of an image using a predetermined kernel
    """
    # make sure it has 5 dimensions
    image = tf.expand_dims(image, 4)

    kernel_size = 3
    p = (kernel_size - 1) // 2
    image = tf.pad(image, [[0,0],[p,p],[p,p], [p,p],[0,0]], 'SYMMETRIC')

    conv = tf.nn.conv3d(image, kernel, strides=[1,1,1,1,1], padding='VALID')

    # remove the extra dimension
    conv = tf.squeeze(conv, 4)
    return conv

def calculate_divergence(u, v, w):
    """
        Calculate divergence for the corresponding velocity component
    """
    kernels = create_divergence_kernels()
    dudx = calculate_gradient(u, kernels[0])
    dvdy = calculate_gradient(v, kernels[1])
    dwdz = calculate_gradient(w, kernels[2])

    return (dudx, dvdy, dwdz)

def calculate_divergence_loss2(u, v, w, u_pred, v_pred, w_pred):
    (divpx, divpy, divpz) = calculate_divergence(u_pred, v_pred, w_pred)
    (divx, divy, divz) = calculate_divergence(u, v, w)
    
    return (divpx - divx) ** 2 + (divpy - divy) ** 2 + (divpz - divz) ** 2

def calculate_relative_error(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5

    u_diff = tf.square(u_pred - u_hi)
    v_diff = tf.square(v_pred - v_hi)
    w_diff = tf.square(w_pred - w_hi)

    diff_speed = tf.sqrt(u_diff + v_diff + w_diff)
    actual_speed = tf.sqrt(tf.square(u_hi) + tf.square(v_hi) + tf.square(w_hi)) 

    # actual speed can be 0, resulting in inf
    relative_speed_loss = diff_speed / (actual_speed + epsilon)
    
    # Make sure the range is between 0 and 1
    relative_speed_loss = tf.clip_by_value(relative_speed_loss, 0., 1.)

    # Apply correction, only use the diff speed if actual speed is zero
    condition = tf.not_equal(actual_speed, tf.constant(0.))
    corrected_speed_loss = tf.where(condition, relative_speed_loss, diff_speed)

    multiplier = 1e4 # round it so we don't get any infinitesimal number
    corrected_speed_loss = tf.round(corrected_speed_loss * multiplier) / multiplier
    # print(corrected_speed_loss)
    
    # Apply mask
    # binary_mask_condition = (mask > threshold)
    binary_mask_condition = tf.equal(binary_mask, 1.0)          
    corrected_speed_loss = tf.where(binary_mask_condition, corrected_speed_loss, tf.zeros_like(corrected_speed_loss))
    # print(found_indexes)

    # Calculate the mean from the total non zero accuracy, divided by the masked area
    # reduce first to the 'batch' axis
    mean_err = tf.reduce_sum(corrected_speed_loss, axis=[1,2,3]) / (tf.reduce_sum(binary_mask, axis=[1,2,3]) + 1) 

    # now take the actual mean
    # mean_err = tf.reduce_mean(mean_err) * 100 # in percentage
    mean_err = mean_err * 100

    return mean_err