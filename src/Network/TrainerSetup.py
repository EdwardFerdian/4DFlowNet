"""
4DFlowNet: Super Resolution ResNet
Author: Edward Ferdian
Date:   14/06/2019
"""

import tensorflow as tf
import numpy as np
import datetime
import time
import shutil
import os
from .SR4DFlowNet import SR4DFlowNet
from . import utility, h5util, loss_utils

class TrainerSetup:
    # constructor
    def __init__(self, patch_size, res_increase, initial_learning_rate=1e-4, quicksave_enable=True, network_name='4DFlowNet', low_resblock=8, hi_resblock=4):
        """
            TrainerSetup constructor
            Setup all the placeholders, network graph, loss functions and optimizer here.
        """
        div_weight = 1e-2 # Weighting for divergenc loss

        # General param
        self.res_increase = res_increase
        hires_patch_size = patch_size * res_increase

        # Network
        self.network_name = network_name

        # Training params
        self.QUICKSAVE_ENABLED = quicksave_enable
        self.iter_per_epoch = 100

        input_shape = (patch_size,patch_size,patch_size,1)

        # input 
        u = tf.keras.layers.Input(shape=input_shape, name='u')
        v = tf.keras.layers.Input(shape=input_shape, name='v')
        w = tf.keras.layers.Input(shape=input_shape, name='w')

        u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
        v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
        w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

        input_layer = [u,v,w,u_mag, v_mag, w_mag]
        net = SR4DFlowNet(res_increase)
        self.predictions = net.build_network(u, v, w, u_mag, v_mag, w_mag, low_resblock, hi_resblock)
        self.model = tf.keras.Model(input_layer, self.predictions)

        u_pred = self.predictions[:,:,:,:,0]
        v_pred = self.predictions[:,:,:,:,1]
        w_pred = self.predictions[:,:,:,:,2]

        # Give identity so we can call them separately
        self.predictions = tf.identity(self.predictions, name="preds")
        self.u_pred = tf.identity(u_pred, name="u_")
        self.v_pred = tf.identity(v_pred, name="v_")
        self.w_pred = tf.identity(w_pred, name="w_")

        # ===== Loss function =====
        self.summary_loss_name = "MSE+div2"
        self.loss_object = tf.keras.losses.MeanSquaredError()

        # MSE
        # speed_diff = self.calculate_mse(self.u_hi, self.v_hi, self.w_hi, self.u_pred, self.v_pred, self.w_pred)
        # mse = tf.reduce_mean(speed_diff)
        # # Divergence loss
        # divergence_loss = loss_utils.calculate_divergence_loss2(self.u_hi, self.v_hi, self.w_hi, self.u_pred, self.v_pred, self.w_pred)
        # div_loss = tf.reduce_mean(divergence_loss)        
        
        # # Calculate total loss
        # self.loss = mse + div_weight * div_loss
        # self.loss = tf.identity(self.loss, name="loss")
        # print(f"Divergence loss2 * {div_weight}")

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
    
        # # Evaluation metric
        # self.rel_loss = loss_utils.calculate_relative_error(self.u_pred, self.v_pred, self.w_pred, self.u_hi, self.v_hi, self.w_hi, self.binary_mask)
        # self.rel_loss = tf.identity(self.rel_loss, name="rel_loss")

        # # Prepare tensorboard summary
        # tf.summary.scalar(f'{self.network_name}/Divergence_loss2', mse)
        # tf.summary.scalar(f'{self.network_name}/MSE'             , div_loss)

        # # learning rate and training optimizer
        self.learning_rate = initial_learning_rate
        
        # # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def adjust_learning_rate(self, epoch):
        if epoch > 0 and epoch % 1 == 0:
            self.optimizer.lr = self.optimizer.lr / np.sqrt(2)
            print(f'Learning rate adjusted to {self.optimizer.lr.numpy():.6f}')

    def calculate_mse(self, u, v, w, u_pred, v_pred, w_pred):
        """
            Calculate Speed magnitude error
        """
        return (u_pred - u) ** 2 +  (v_pred - v) ** 2 + (w_pred - w) ** 2

    def init_model_dir(self):
        """
            Create model directory to save the weights with a [network_name]_[datetime] format
            Also prepare logfile and tensorboard summary within the directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = f'{self.network_name}_{timestamp}'

        self.model_dir = f"../models/{self.unique_model_name}"
        # Do not use .ckpt on the model_path
        self.model_path = f"{self.model_dir}/{self.network_name}"

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        # summary - Tensorboard stuff
        self._prepare_logfile_and_summary()
    
    def _prepare_logfile_and_summary(self):
        """
            Prepare csv logfile to keep track of the loss and Tensorboard summaries
        """
        # summary - Tensorboard stuff
        self.train_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/train')
        self.val_writer = tf.summary.create_file_writer(self.model_dir+'/tensorboard/validate')

        # Prepare log file
        self.logfile = self.model_dir + '/loss.csv'

        utility.log_to_file(self.logfile, f'Network: {self.network_name}\n')
        utility.log_to_file(self.logfile, f'learning rate: {self.learning_rate}\n')
        utility.log_to_file(self.logfile, f'epoch, train_err, val_err, train_rel_err, val_rel_err, best_model, benchmark_err, benchmark_rel_err\n')

        print("Copying source code to model directory...")
        # Copy all the source file to the model dir for backup
        directory_to_backup = [".", "Network"]
        for directory in directory_to_backup:
            files = os.listdir(directory)
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".ipynb"):
                    dest_fpath = os.path.join(self.model_dir,"source",directory, fname)
                    os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)

                    shutil.copy2(f"{directory}/{fname}", dest_fpath)

    def _update_summary_logging(self, epoch):
        """
            Tf.summary for epoch level loss
        """
        # Summary writer
        with self.train_writer.as_default():
            # other model code would go here
            tf.summary.scalar("loss", self.train_loss.result(), step=epoch)
            tf.summary.scalar("rel_loss", self.train_accuracy.result(), step=epoch)
            # tf.summary.scalar("learning_rate", self.optimizer.lr, step=epoch)
        
        with self.val_writer.as_default():
            # other model code would go here
            tf.summary.scalar("loss", self.val_loss.result(), step=epoch)
            tf.summary.scalar("rel_loss", self.val_accuracy.result(), step=epoch)
        
    def quicksave(self, testset, epoch_nr):
        """
            Predict a batch of data from the benchmark_iterator.
            This is saved under the model directory with the name quicksave_[network_name].h5
            Quicksave is done everytime the best model is saved.
        """
        for i, (data_pairs) in enumerate(testset):
            u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
            hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
            input_data = [u,v,w, u_mag, v_mag, w_mag]

            preds = self.model.predict(input_data)

            # TODO:
            loss_val = self.loss_object(hires, preds)
            rel_loss = 0
            # Do only 1 batch
            break

        quicksave_filename = f"quicksave_{self.network_name}.h5"
        h5util.save_predictions(self.model_dir, quicksave_filename, "epoch", np.asarray([epoch_nr]), compression='gzip')

        preds = np.expand_dims(preds, 0) # Expand dim to [epoch_nr, batch, ....]
        h5util.save_predictions(self.model_dir, quicksave_filename, "u", preds[:, :,:,:,:, 0], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "v", preds[:, :,:,:,:, 1], compression='gzip')
        h5util.save_predictions(self.model_dir, quicksave_filename, "w", preds[:, :,:,:,:, 2], compression='gzip')

        if epoch_nr == 1:
            # Save the actual data only for the first epoch
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_u", u, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_v", v, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "lr_w", w, compression='gzip')

            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_u", np.squeeze(u_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_v", np.squeeze(v_hr, -1), compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "hr_w", np.squeeze(w_hr, -1), compression='gzip')
            
            h5util.save_predictions(self.model_dir, quicksave_filename, "venc", venc, compression='gzip')
            h5util.save_predictions(self.model_dir, quicksave_filename, "mask", np.squeeze(mask, -1), compression='gzip')
        
        return loss_val, rel_loss
      
    @tf.function
    def train_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            input_data = [u,v,w, u_mag, v_mag, w_mag]
            predictions = self.model(input_data, training=True)
            loss = self.loss_object(hires, predictions)

        # Get the gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Update the weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update the loss and accuracy
        self.train_loss(loss)
        self.train_accuracy(hires, predictions)

        # logging batch loss
        with self.train_writer.as_default():
            tf.summary.scalar("batch_loss", loss, step=self.optimizer.iterations)

    @tf.function
    def test_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        input_data = [u,v,w, u_mag, v_mag, w_mag]
        predictions = self.model(input_data, training=False)
        t_loss = self.loss_object(hires, predictions)
        

        self.val_loss(t_loss)
        self.val_accuracy(hires, predictions)
        return predictions

    def train_network(self, trainset, valset, n_epoch, testset=None):
        """
            Main training function. Receives trainining and validation dataset iterator.
            When validation iterator is None, the training loss/accuracty is used for saved criteria instead.
        """
        # ----- Run the training -----
        print("==================== TRAINING =================")
        print(f'Learning rate {self.optimizer.lr.numpy():.7f}')
        print(f"Start training at {time.ctime()} - {self.unique_model_name}\n")
        start_time = time.time()
        
        # Setup acc and data count
        previous_loss = np.inf
        total_batch_train = tf.data.experimental.cardinality(trainset).numpy()
        total_batch_val = tf.data.experimental.cardinality(valset).numpy()

        for epoch in range(n_epoch):
            # ------------------------------- Training -------------------------------
            self.adjust_learning_rate(epoch)

            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            start_loop = time.time()
            
             # --- Training ---
            for i, (data_pairs) in enumerate(trainset):
                self.train_step(data_pairs)
                message = f"Epoch {epoch+1} Train batch {i+1}/{total_batch_train} | loss: {self.train_loss.result():.5f} ({self.train_accuracy.result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')


            for i, (data_pairs) in enumerate(valset):
                self.test_step(data_pairs)
                message = f"Epoch {epoch+1} Validation batch {i+1}/{total_batch_val} | loss: {self.val_loss.result():.5f} ({self.val_accuracy.result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')


            message = f'\rEpoch {epoch+1} Train loss: {self.train_loss.result():.5f} ({self.train_accuracy.result():.1f} %), Val loss: {self.val_loss.result():.5f} ({self.val_accuracy.result():.1f} %) - {time.time()-start_loop:.1f} secs'
            log_line = f'{epoch+1},{self.train_loss.result():.7f},{self.val_loss.result():.7f},{self.train_accuracy.result():.2f}%,{self.val_accuracy.result():.2f}%'

            self._update_summary_logging(epoch)

            # # TODO: Save criteria
            if self.val_accuracy.result() < previous_loss:
                self.model.save(f'{self.model_path}_weights.h5')
                
                # Update best acc
                previous_loss = self.val_accuracy.result()
                
                # logging
                message  += ' **' # Mark as saved
                log_line += ',**'

                if self.QUICKSAVE_ENABLED and testset is not None:
                    quick_loss, quick_accuracy = self.quicksave(testset, epoch+1)
                    message  += f' Benchmark loss: {quick_loss:.5f} ({quick_accuracy:.1f} %)'
                    log_line += f', {quick_loss:.7f}, {quick_accuracy:.2f}%'

            print(message)
            utility.log_to_file(self.logfile, log_line+"\n")
            # /END of epoch loop

        print(f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}")
        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        print(f"Total training time: {hrs} hrs {mins} mins {secs} secs.")
        print(f"Finished at {time.ctime()}")
        print("==================== END TRAINING =================")