"""
4DFlowNet: Super Resolution ResNet
Author: Edward Ferdian
Date:   14/06/2019
"""
import pickle
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
        self.div_weight = 1e-2 # Weighting for divergence loss

        # General param
        self.res_increase = res_increase
        
        # Training params
        self.QUICKSAVE_ENABLED = quicksave_enable
        
        # Network
        self.network_name = network_name

        input_shape = (patch_size,patch_size,patch_size,1)

        # Prepare Input 
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

        # ===== Loss function =====
        print(f"Divergence loss2 * {self.div_weight}")

        self.loss_metrics = dict([
            ('train_loss', tf.keras.metrics.Mean(name='train_loss')),
            ('val_loss', tf.keras.metrics.Mean(name='val_loss')),
            ('train_accuracy', tf.keras.metrics.Mean(name='train_accuracy')),
            ('val_accuracy', tf.keras.metrics.Mean(name='val_accuracy')),
            ('train_mse', tf.keras.metrics.Mean(name='train_mse')),
            ('val_mse', tf.keras.metrics.Mean(name='val_mse')),
            ('train_div', tf.keras.metrics.Mean(name='train_div')),
            ('val_div', tf.keras.metrics.Mean(name='val_div')),

            ('l2_reg_loss', tf.keras.metrics.Mean(name='l2_reg_loss')),
        ])
        self.accuracy_metric = 'val_loss'

        # learning rate and training optimizer
        self.learning_rate = initial_learning_rate
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        
        # Compile model so we can save the optimizer weights
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)

    def adjust_learning_rate(self, epoch):
        # For 14k rows of data and batch 20, this is ~10k iterations
        if epoch > 0 and epoch % 14 == 0:
            self.optimizer.lr = self.optimizer.lr / np.sqrt(2)
            message = f'Learning rate adjusted to {self.optimizer.lr.numpy():.6f} - {time.ctime()}\n'
            print(message)

    def loss_function(self, y_true, y_pred):
        """
            Calculate Total Loss function
            Loss = MSE + weight * div_loss2
        """
        u,v,w = y_true[:,:,:,:,0],y_true[:,:,:,:,1], y_true[:,:,:,:,2]
        u_pred,v_pred,w_pred = y_pred[:,:,:,:,0],y_pred[:,:,:,:,1], y_pred[:,:,:,:,2]

        mse = self.calculate_mse(u,v,w, u_pred,v_pred,w_pred)
        divergence_loss = loss_utils.calculate_divergence_loss2(u,v,w, u_pred,v_pred,w_pred)
        
        return  tf.reduce_mean(mse) + self.div_weight * tf.reduce_mean(divergence_loss)

    def accuracy_function(self, y_true, y_pred, mask):
        """
            Calculate relative speed error
        """
        u,v,w = y_true[:,:,:,:,0],y_true[:,:,:,:,1], y_true[:,:,:,:,2]
        u_pred,v_pred,w_pred = y_pred[:,:,:,:,0],y_pred[:,:,:,:,1], y_pred[:,:,:,:,2]

        return loss_utils.calculate_relative_error(u_pred, v_pred, w_pred, u, v, w, mask)

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
        utility.log_to_file(self.logfile, f'Initial learning rate: {self.learning_rate}\n')
        utility.log_to_file(self.logfile, f'Accuracy metric: {self.accuracy_metric}\n')
        utility.log_to_file(self.logfile, f'Divergence weight: {self.div_weight}\n')

        # Header
        stat_names = ','.join(self.loss_metrics.keys()) # train and val stat names
        utility.log_to_file(self.logfile, f'epoch, {stat_names}, learning rate, elapsed (sec), best_model, benchmark_err, benchmark_rel_err, benchmark_mse, benchmark_divloss\n')

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

      
    @tf.function
    def train_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            input_data = [u,v,w, u_mag, v_mag, w_mag]
            predictions = self.model(input_data, training=True)
            loss = self.loss_function(hires, predictions)

        # Get the gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Update the weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update the loss and accuracy
        self.train_loss(loss)

        rel_error = self.accuracy_function(hires, predictions, mask)
        self.train_accuracy(rel_error)
        # self.train_accuracy(hires, predictions)

        # logging batch loss
        # tf.print("\nBatch loss check", loss, self.train_loss.result())
        # tf.print("\nBatch acc check", rel_error, self.train_accuracy.result())
        with self.train_writer.as_default():
            tf.summary.scalar(f"{self.network_name}/batch_loss", loss, step=self.optimizer.iterations)
            tf.summary.scalar(f"{self.network_name}/batch_acc" , rel_error, step=self.optimizer.iterations)

    @tf.function
    def test_step(self, data_pairs):
        u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
        hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        input_data = [u,v,w, u_mag, v_mag, w_mag]
        predictions = self.model(input_data, training=False)
        
        self.calculate_and_update_metrics(hires, predictions, mask, 'val')
       

        return predictions

    def calculate_and_update_metrics(self, hires, predictions, mask, metric_set):
        loss, mse, divloss = self.loss_function(hires, predictions, mask)
        rel_error = self.accuracy_function(hires, predictions, mask)
        
        if metric_set == 'train':
            l2_reg_loss = self.calculate_regularizer_loss()
            self.loss_metrics[f'l2_reg_loss'].update_state(l2_reg_loss)

            loss += l2_reg_loss

        # Update the loss and accuracy
        self.loss_metrics[f'{metric_set}_loss'].update_state(loss)

        self.loss_metrics[f'{metric_set}_mse'].update_state(mse)
        self.loss_metrics[f'{metric_set}_div'].update_state(divloss)
        self.loss_metrics[f'{metric_set}_accuracy'].update_state(rel_error)

        # self.loss_metrics[f'{metric_set}_wall_loss'].update_state(wall_loss)
        # self.loss_metrics[f'{metric_set}_nonwall_loss'].update_state(nonwall_loss)

        return loss

    def reset_metrics(self):
        for key in self.loss_metrics.keys():
            self.loss_metrics[key].reset_states()


    def train_network(self, trainset, valset, n_epoch, testset=None):
        """
            Main training function. Receives trainining and validation TF dataset.
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
                # Train the network
                self.train_step(data_pairs)
                message = f"Epoch {epoch+1} Train batch {i+1}/{total_batch_train} | loss: {self.train_loss.result():.5f} ({self.train_accuracy.result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Validation ---
            for i, (data_pairs) in enumerate(valset):
                self.test_step(data_pairs)
                message = f"Epoch {epoch+1} Validation batch {i+1}/{total_batch_val} | loss: {self.val_loss.result():.5f} ({self.val_accuracy.result():.1f} %) - {time.time()-start_loop:.1f} secs"
                print(f"\r{message}", end='')

            # --- Epoch logging ---
            message = f'\rEpoch {epoch+1} Train loss: {self.train_loss.result():.5f} ({self.train_accuracy.result():.1f} %), Val loss: {self.val_loss.result():.5f} ({self.val_accuracy.result():.1f} %) - {time.time()-start_loop:.1f} secs'
            log_line = f'{epoch+1},{self.train_loss.result():.7f},{self.val_loss.result():.7f},{self.train_accuracy.result():.2f}%,{self.val_accuracy.result():.2f}%,{self.optimizer.lr.numpy():.6f},{time.time()-start_loop:.1f}'
            self._update_summary_logging(epoch)

            # --- Save criteria ---
            if self.val_accuracy.result() < previous_loss:
                self.model.save(f'{self.model_path}_weights.h5')
                
                # Update best acc
                previous_loss = self.val_accuracy.result()
                
                # logging
                message  += ' **' # Mark as saved
                log_line += ',**'

                # Benchmarking
                if self.QUICKSAVE_ENABLED and testset is not None:
                    quick_loss, quick_accuracy = self.quicksave(testset, epoch+1)
                    message  += f' Benchmark loss: {quick_loss:.5f} ({quick_accuracy:.1f} %)'
                    log_line += f', {quick_loss:.7f}, {quick_accuracy:.2f}%'
            # Logging
            print(message)
            utility.log_to_file(self.logfile, log_line+"\n")
            # /END of epoch loop

        # End
        hrs, mins, secs = utility.calculate_time_elapsed(start_time)
        message =  f"\nTraining {self.network_name} completed! - name: {self.unique_model_name}"
        message += f"\nTotal training time: {hrs} hrs {mins} mins {secs} secs."
        message += f"\nFinished at {time.ctime()}"
        message += f"\n==================== END TRAINING ================="
        utility.log_to_file(self.logfile, message)
        print(message)
        
        # Finish!
        
    def save_best_model(self):
        """
            Save model weights and also optmizer weights to enable restore model
            to continue training

            Based on:
            https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        """
        # Save model weights.
        self.model.save(f'{self.model_path}-best.h5')
        
        # Save optimizer weights.
        symbolic_weights = getattr(self.optimizer, 'weights')
        if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(f'{self.model_dir}/optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)

    def restore_model(self, old_model_dir, old_model_file):
        """
            Restore model weights and optimizer weights for uncompiled model
            Based on: https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state

            For an uncompiled model, we cannot just set the optmizer weights directly because they are zero.
            We need to at least do an apply_gradients once and then set the optimizer weights.
        """
        # Set the path for the weights and optimizer
        model_weights_path = f"{old_model_dir}/{old_model_file}"
        opt_path   = f"{old_model_dir}/optimizer.pkl"

        # Load the optimizer weights
        with open(opt_path, 'rb') as f:
            opt_weights = pickle.load(f)
        
        # Get the model's trainable weights
        grad_vars = self.model.trainable_weights
        # This need not be model.trainable_weights; it must be a correctly-ordered list of 
        # grad_vars corresponding to how you usually call the optimizer.
        zero_grads = [tf.zeros_like(w) for w in grad_vars]

        # Apply gradients which don't do nothing with Adam
        self.optimizer.apply_gradients(zip(zero_grads, grad_vars))

        # Set the weights of the optimizer
        self.optimizer.set_weights(opt_weights)

        # NOW set the trainable weights of the model
        self.model.load_weights(model_weights_path)

    def _update_summary_logging(self, epoch):
        """
            Tf.summary for epoch level loss
        """
        # Filter out the train and val metrics
        train_metrics = {k.replace('train_',''): v for k, v in self.loss_metrics.items() if k.startswith('train_')}
        val_metrics = {k.replace('val_',''): v for k, v in self.loss_metrics.items() if k.startswith('val_')}
        
        # Summary writer
        with self.train_writer.as_default():
            tf.summary.scalar(f"{self.network_name}/learning_rate", self.optimizer.lr, step=epoch)
            for key in train_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  train_metrics[key].result(), step=epoch)         
        
        with self.val_writer.as_default():
            for key in val_metrics.keys():
                tf.summary.scalar(f"{self.network_name}/{key}",  val_metrics[key].result(), step=epoch)
   
        
    def quicksave(self, testset, epoch_nr):
        """
            Predict a batch of data from the benchmark testset.
            This is saved under the model directory with the name quicksave_[network_name].h5
            Quicksave is done everytime the best model is saved.
        """
        for i, (data_pairs) in enumerate(testset):
            u,v,w, u_mag, v_mag, w_mag, u_hr,v_hr, w_hr, venc, mask = data_pairs
            hires = tf.concat((u_hr, v_hr, w_hr), axis=-1)
            input_data = [u,v,w, u_mag, v_mag, w_mag]

            preds = self.model.predict(input_data)

            loss_val = self.loss_function(hires, preds)
            rel_loss = self.accuracy_function(hires, preds, mask)
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
            h5util.save_predictions(self.model_dir, quicksave_filename, "mask", mask, compression='gzip')
        
        return loss_val, rel_loss