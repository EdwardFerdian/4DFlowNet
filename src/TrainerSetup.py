"""
4DFlowNet: Super Resolution ResNet
Author: Edward Ferdian
Date:   14/06/2019
"""

import tensorflow as tf
import numpy as np
import time
import datetime
import utility
import h5util
import loss_utils
import os
import shutil
import SR4DFlowNet as network

class TrainerSetup:
    # constructor
    def __init__(self, patch_size, res_increase, initial_learning_rate=1e-4, quicksave_enable=True, network_name='4DFlowNet'):
        
        div_weight = 1e-2

        self.network_name = network_name
        self.QUICKSAVE_ENABLED = quicksave_enable
        
        self.iter_per_epoch = 100
        self.overall_data_read = 0
        self.iteration_count = 0

        self.res_increase = res_increase
        hires_patch_size = patch_size * res_increase
        
        self.sess = tf.Session()

        # Placeholders
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # input 
        self.u = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="u")
        self.v = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="v")
        self.w = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="w")
        
        self.u_mag = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="u_mag")
        self.v_mag = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="v_mag")
        self.w_mag = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, patch_size], name="w_mag")

        # label
        self.u_hi = tf.placeholder(tf.float32, shape=[None, hires_patch_size, hires_patch_size, hires_patch_size], name="u_hi")
        self.v_hi = tf.placeholder(tf.float32, shape=[None, hires_patch_size, hires_patch_size, hires_patch_size], name="v_hi")
        self.w_hi = tf.placeholder(tf.float32, shape=[None, hires_patch_size, hires_patch_size, hires_patch_size], name="w_hi")
        self.binary_mask = tf.placeholder(tf.float32, shape=[None, hires_patch_size, hires_patch_size, hires_patch_size], name="binary_mask")

        u = tf.expand_dims(self.u, 4)
        v = tf.expand_dims(self.v, 4)
        w = tf.expand_dims(self.w, 4)

        u_mag = tf.expand_dims(self.u_mag, 4)
        v_mag = tf.expand_dims(self.v_mag, 4)
        w_mag = tf.expand_dims(self.w_mag, 4)

        # network & output
        net = network.SR4DFlowNet(res_increase)
        self.predictions = net.build_network(u, v, w, u_mag, v_mag, w_mag)
        
        u_pred = self.predictions[:,:,:,:,0]
        v_pred = self.predictions[:,:,:,:,1]
        w_pred = self.predictions[:,:,:,:,2]

        self.predictions = tf.identity(self.predictions, name="preds")
        self.u_pred = tf.identity(u_pred, name="u_")
        self.v_pred = tf.identity(v_pred, name="v_")
        self.w_pred = tf.identity(w_pred, name="w_")

        # Loss function
        speed_diff = self.calculate_mse(self.u_hi, self.v_hi, self.w_hi, self.u_pred, self.v_pred, self.w_pred)

        # add divergence loss
        self.summary_loss_name = "MSE+div2"
        divergence_loss = loss_utils.calculate_divergence_loss2(self.u_hi, self.v_hi, self.w_hi, self.u_pred, self.v_pred, self.w_pred)
        
        print(f"Divergence loss2 * {div_weight}")
        
        # calculate total loss
        mse = tf.reduce_mean(speed_diff)
        div_loss = tf.reduce_mean(divergence_loss)
        self.loss = mse + div_weight * div_loss
        self.loss = tf.identity(self.loss, name="loss")
        
        # Evaluation metric
        self.rel_loss = loss_utils.calculate_relative_error(self.u_pred, self.v_pred, self.w_pred, self.u_hi, self.v_hi, self.w_hi, self.binary_mask)
        self.rel_loss = tf.identity(self.rel_loss, name="rel_loss")

        tf.summary.scalar(f'{self.network_name}/Divergence_loss2', mse)
        tf.summary.scalar(f'{self.network_name}/MSE'             , div_loss)

        # learning rate and training optimizer
        self.learning_rate = tf.Variable( initial_value = initial_learning_rate, trainable = False, name = 'learning_rate' ) 
        self.adjust_learning_rate = tf.assign(self.learning_rate, self.learning_rate / tf.sqrt(2.))
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss)

        # initialise the variables
        print("Initializing session...")
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    def calculate_mse(self, u, v, w, u_pred, v_pred, w_pred):
        return (u_pred - u) ** 2 +  (v_pred - v) ** 2 + (w_pred - w) ** 2

    def init_model_dir(self):        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f'{self.network_name}_{timestamp}'

        self.model_dir = f"../models/{self.unique_model_name}"
        # Do not use .ckpt on the model_path
        self.model_path = f"{self.model_dir}/{self.network_name}"

        # summary - Tensorboard stuff
        self._prepare_loss_and_summary()

        # Copy files to the log dir
        # shutil.copy2('Res3Dsr.py', os.path.join(self.model_dir, 'Res3Dsr.py'))
        # shutil.copy2('loss_utils.py', os.path.join(self.model_dir, 'loss_utils.py'))
        # shutil.copy2('zoomresmag3D_trainer.ipynb', os.path.join(self.model_dir, 'trainer.ipynb'))
        # shutil.copy2('PatchHandler3D.py', os.path.join(self.model_dir, 'PatchHandler3D.py'))
        return self.sess
    
    def _prepare_loss_and_summary(self):
        # summary - Tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.model_dir+'/tensorboard/train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.model_dir+'/tensorboard/validate', self.sess.graph)
        # Prepare loss file
        self.logfile = self.model_dir + '/loss.csv'

        utility.log_to_file(self.logfile, f'Network: {self.network_name}\n')
        utility.log_to_file(self.logfile, f'learning rate: {self.learning_rate}\n')
        # utility.log_to_file(self.logfile, f'batch size: {self.batch_size}\n')
        utility.log_to_file(self.logfile, f'epoch, train_err, val_err, train_rel_err, val_rel_err, best_model\n')

    def _update_summary_logging(self, epoch, epoch_loss, epoch_relloss, is_training):
        """
            Tf.summary for epoch level loss
        """
        summary = tf.Summary()
        summary.value.add(tag=f'{self.unique_model_name}/Loss ({self.summary_loss_name})', simple_value=epoch_loss)
        summary.value.add(tag=f'{self.unique_model_name}/RelLoss', simple_value=epoch_relloss)

        if is_training:
            self.train_writer.add_summary(summary, epoch)
        else:
            self.val_writer.add_summary(summary, epoch)
    
    def restore_model(self, model_dir, model_name):
        print(f'Restoring model {model_name}')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        
    def quicksave(self, val_iterator, next_batch, epoch_nr):
        # log.info(f'Saving prediction on epoch {epoch_nr}')

        for i in range(2):
            next_element = self.sess.run(next_batch)

            batch_u, batch_uhr, batch_umag = next_element[0], next_element[1] , next_element[2]  # LR, HR, MAG
            batch_v, batch_vhr, batch_vmag = next_element[3], next_element[4] , next_element[5]  
            batch_w, batch_whr, batch_wmag = next_element[6], next_element[7] , next_element[8]  
            batch_venc, batch_mask = next_element[9], next_element[10]

            tf_dict = { self.u: batch_u, self.u_hi: batch_uhr, self.u_mag: batch_umag, 
                        self.v: batch_v, self.v_hi: batch_vhr, self.v_mag: batch_vmag, 
                        self.w: batch_w, self.w_hi: batch_whr, self.w_mag: batch_wmag, 
                        self.is_training: False, self.binary_mask: batch_mask }
            
            var_to_run = [self.loss, self.rel_loss, self.predictions]
            loss_val, rel_loss, preds = self.sess.run(var_to_run, feed_dict=tf_dict)

            quicksave_filename = f"quicksave_{self.network_name}.h5"
            h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression='gzip')

            h5util.save_predictions(self.model_dir, quicksave_filename, "u", preds[:,:,:,:, 0], compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "v", preds[:,:,:,:, 1], compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "w", preds[:,:,:,:, 2], compression='gzip')

            # log.info(f"Quicksave Loss: {loss_val:.5f} | Rel_err: {rel_loss:.2f}%")

            if epoch_nr == 1:
                h5util.save_predictions(self.model_dir, quicksave_filename, "lr_u", batch_u, compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "lr_v", batch_v, compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "lr_w", batch_w, compression='gzip')

                h5util.save_predictions(self.model_dir, quicksave_filename, "hr_u", batch_uhr, compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "hr_v", batch_vhr, compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "hr_w", batch_whr, compression='gzip')
                
                h5util.save_predictions(self.model_dir, quicksave_filename, "venc", batch_venc, compression='gzip')
                h5util.save_predictions(self.model_dir, quicksave_filename, "mask", batch_mask, compression='gzip')
        
    def run_batch(self, next_batch, is_training):
        batch_u, batch_uhr, batch_umag = next_batch[0], next_batch[1] , next_batch[2]  # LR, HR, MAG
        batch_v, batch_vhr, batch_vmag = next_batch[3], next_batch[4] , next_batch[5]  
        batch_w, batch_whr, batch_wmag = next_batch[6], next_batch[7] , next_batch[8]  
        batch_mask = next_batch[10]

        tf_dict = { self.u: batch_u, self.u_hi: batch_uhr, self.u_mag: batch_umag, 
                    self.v: batch_v, self.v_hi: batch_vhr, self.v_mag: batch_vmag, 
                    self.w: batch_w, self.w_hi: batch_whr, self.w_mag: batch_wmag, 
                    self.is_training: is_training, self.binary_mask: batch_mask }

        if is_training:
            var_to_run = [self.train_op, self.merged, self.loss, self.rel_loss]
            _, summary, loss_value, rel_loss = self.sess.run(var_to_run, feed_dict=tf_dict)
            # For training, add it to summary directly on batch level
            self.train_writer.add_summary(summary, self.overall_data_read)
        else:
            var_to_run = [self.loss, self.rel_loss]
            loss_value, rel_loss = self.sess.run(var_to_run, feed_dict=tf_dict)
        
        return loss_value, rel_loss

    def predict(self, u, v, w, u_mag, v_mag, w_mag):
        feed_dict = { self.u: u, self.v: v, self.w: w,
                      self.u_mag: u_mag, self.v_mag: v_mag, self.w_mag: w_mag }
        srImages = self.sess.run(self.predictions, feed_dict)
        
        return srImages

    def train_network(self, train_iterator, val_iterator, n_epoch):
        # ----- Run the training -----
        lr = self.sess.run(self.learning_rate)

        print("==================== TRAINING =================")
        print(f'Learning rate {lr:.7f}')
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n")
        start_time = time.time()
        
        next_element = train_iterator.get_next()
        next_val = val_iterator.get_next()

        self.sess.run(train_iterator.initializer)
        self.sess.run(val_iterator.initializer)

        previous_loss = np.inf
        for epoch in range(n_epoch):
            # ------------------------------- Training -------------------------------
            # Reduce learning rate every few steps
            if epoch >= 5 and epoch % 30 == 0:
            # if self.iteration_count > 0 and self.iteration_count % 0 == 0:
                self.adjust_learning_rate.eval(session=self.sess)
                lr = self.sess.run(self.learning_rate)
                print(f'Learning rate adjusted to {lr:.7f}')
                # self.iteration_count = 0

            start_loop = time.time()
            # Training
            train_loss, train_relloss = self.feed_all_batches(train_iterator, next_element, epoch, is_training = True)

            # Validation
            if val_iterator is not None:
                # Validation step
                val_loss, val_relloss = self.feed_all_batches(val_iterator, next_val, epoch, is_training = False)
                avg_loss = val_loss
                avg_relloss = val_relloss
            else:
                # There is no validation set, just use the train loss and "acc"
                avg_loss = train_loss
                avg_relloss = train_relloss
                val_loss = 0 # no validation set

            message = f'\rEpoch {epoch+1}, Train loss: {train_loss:.5f} ({train_relloss:.1f} %), Val loss: {val_loss:.5f} ({val_relloss:.1f} %) - {time.time()-start_loop:.2f} secs'
            log_line = f'{epoch+1},{train_loss:.7f},{val_loss:.7f},{train_relloss:.2f}%,{val_relloss:.2f}%'
            # Save criteria 
            if avg_relloss < previous_loss:
                save_path = self.saver.save(self.sess, self.model_path)
                # print("\nModel saved in: %s" % save_path)
                
                # Update best acc
                previous_loss = avg_relloss
                message +=  ' **' # Mark as saved
                log_line += ',**'

                if self.QUICKSAVE_ENABLED:
                    # self.benchmark = Benchmark(self.benchmark_path, self.model_dir, self.network_name, self.batch_size)
                    self.quicksave(val_iterator, next_val, epoch+1)
                    self.sess.run(val_iterator.initializer) # reset iterator

            print(message)
            utility.log_to_file(self.logfile, log_line+"\n")
            # /END of epoch loop

        print(f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}")
        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        print(f"Total training time: {hrs} hrs {mins} mins {secs} secs.")
        print(f"Finished at {time.ctime()}")
        print("==================== END TRAINING =================")

    def feed_all_batches(self, data_iterator, next_element, epoch, is_training):
        start_loop = time.time()

        total_loss = []
        total_relloss = []
        total_data = 0

        read_all_data = True
        if is_training:
            read_all_data = False

        try:
            i = 0
            # --- run all batch in dataset
            # while True:
            while i < self.iter_per_epoch or read_all_data:
                # we can use a while loop because the iterator is exhaustive
                next_batch = self.sess.run(next_element)
                # batch_loss, batch_relloss, n_per_batch = self.run_batch(next_batch, is_training=is_training)
                batch_loss, batch_relloss, n_per_batch = i, i, 20

                # we do this because n_per_batch in the last batch may be different
                total_loss.append(batch_loss)
                total_relloss.append(batch_relloss)

                total_data    += n_per_batch
                # print(self.train.global_step)

                if is_training:
                    message = f"Epoch {epoch+1} Train batch {i+1}/{self.iter_per_epoch} | loss {batch_loss:.5f} ({batch_relloss:.1f} %) | Elapsed: {time.time()-start_loop:.2f} secs."
                    print(f"\r{message}", end='')
                    # print("\rEpoch %d Train batch %d/%d | loss %.5f (%.2f %) | Elapsed: %.2f secs." % (epoch+1, i+1, self.iter_per_epoch, batch_loss, batch_relloss, time.time()-start_loop), end='')

                    # print ("\rRead %d rows: [%-25s], batch loss %.5f (%.2f) | Elapsed: %.2f secs." % (total_data, '='*(i), batch_loss, batch_relloss, time.time()-start_loop), end='')
                    self.overall_data_read += n_per_batch # increase per number of batch trained
                    self.iteration_count += 1
                else:
                    message = f"Epoch {epoch+1} Val batch {i+1}/{self.iter_per_epoch} | loss {batch_loss:.5f} ({batch_relloss:.1f} %) | Elapsed: {time.time()-start_loop:.2f} secs."
                    print(f"\r{message}", end='')
                    # print("\rEpoch %d Val batch %d/%d | loss %.5f (%.2f %) | Elapsed: %.2f secs." % (epoch+1, i+1, self.iter_per_epoch, batch_loss, batch_relloss, time.time()-start_loop), end='')
                    # print ("\rEval %d rows: [%-25s], batch loss %.5f (%.2f) | Elapsed: %.2f secs." % (total_data, '='*(i), batch_loss, batch_relloss, time.time()-start_loop), end='')
                i+=1 

        except tf.errors.OutOfRangeError:
            # print("\nData iterator has been exhausted, re-initializing...")
            self.sess.run(data_iterator.initializer)

            # Without .repeat(), iterator is exhaustive. This is a common practice
            # If we want to use repeat, then we need to specify the number of batch
            pass

        # calculate the loss per epoch
        epoch_loss = np.mean(total_loss)
        epoch_relloss = np.mean(total_relloss)

        self._update_summary_logging(epoch, epoch_loss, epoch_relloss, is_training)
        
        return epoch_loss, epoch_relloss