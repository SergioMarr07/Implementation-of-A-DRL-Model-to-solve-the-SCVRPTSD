import torch
from logging import getLogger # Managing log messages
from torch.optim import Adam as Optimizer # Adam (a variant of Gradient Descent Algorithm) is the algorithm used to update the NN parameters
from torch.optim.lr_scheduler import MultiStepLR as Scheduler # Scheduler will be use to decrese learning rate at certain training points

class CVRPTrainer:
    def __init__(self,
                 env_params, # A dict with hiperparameters to set up the environment model
                 model_params, # A dict with hiperparameters to set up the NN model
                 optimizer_params, # A dict with hiperparameters to set up the optimizer (Adam algorithm)
                 trainer_params): # A dict with hiperparameters to set up the training process, it could include a checkpoint to restart it

        # Save arguments of the function as attributes of the class
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer') # Instantiation of a getLogger object, it is called trainer
        self.result_folder = get_result_folder() # Returning the path to save the log file, get_result ... function comes from the utils script
        self.result_log = LogData() # An instantiation of a LogData object (it is used to save log information, metrics of training) ... LogData() class comes from the utils script

        # Setting up GPU
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA: # If there's a GPU available
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')

        # Main components
        self.model = CVRPModel(**self.model_params) # NN-model
        self.env = CVRPEnv(**self.env_params) # Environment model
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer']) # Optimizer
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler']) # Scheduler
        self.start_epoch = 1

        # Code block to restore training from a checkpoint
        model_load = trainer_params['model_load'] # See trainer_params to check hiperparameters (enable, path, epoch ...) to set up the checkpoint
        if model_load['enable']: # Binary to know if the training will be restarted from a checkpoint
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device) # Load checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict']) # Setting NN parameters based on the checkpoint
            self.start_epoch = 1 + model_load['epoch'] # Setting epoch num from training should start
            self.result_log.set_raw_data(checkpoint['result_log']) # Updating log file based on previous training session
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Setting optimizer state
            self.scheduler.last_epoch = model_load['epoch']-1 # Setting scheduler state
            self.logger.info('Saved Model Loaded !!') # Printting message in log file and output

        self.time_estimator = TimeEstimator() # Instantation of a TimeEstimator object, to log time execution information, TimeStimator class comes from utils script

    def run(self): # Main method of the class, run simulations and train the NNM
        self.time_estimator.reset(self.start_epoch) # Reset counter for time
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1): # Loop over the total number of epochs
            self.logger.info('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch) # Perform training for one epoch
            self.result_log.append('train_score', epoch, train_score) # Saving metric of train score
            self.result_log.append('train_loss', epoch, train_loss) # Saving metric of train loss

            # LR Decay
            self.scheduler.step() # Updates counter to change lr according to the scheduler

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str)) # Log epoch and time information

            all_done = (epoch == self.trainer_params['epochs']) # If the last epoch occurs, the variable all_done turns to True
            model_save_interval = self.trainer_params['logging']['model_save_interval'] # It indicates the iteration to save the checkpoint

            # Save checkpoint
            if all_done or (epoch % model_save_interval) == 0: # Save only if the conditions are satisfied
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch)) # the checkpoint is storaged within the result folder and with an indicator of the epoch when it is saved

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        # Instantiation of AverageMeter objects from the utilities modules to follow up on training metrics
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode: # Loop over the batches (an episode is used to refer when processing a batch)

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining) # With this the running is not stopped although the number of samples is not a multilple of batches

            avg_score, avg_loss = self._train_one_batch(batch_size) # Perform training for one batch
            # Update metrics
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size # The variable episode will correspond to the number of processed batches

            # Log first 10 batch, first and every 3 epochs
            if epoch == self.start_epoch or epoch % 3 == 0:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg # Return the average of the training metrics

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train() # Set the model in the train mode (actually this is the standard functionality)
        self.env.load_problems(batch_size) # Generating instances of the problems
        reset_state, _, _ = self.env.reset() # Setting up the environment
        self.model.pre_forward(reset_state) # Computing node embeddings with the encoder

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0)) # Global prob tensor
        # shape: (batch, pomo, 0~)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done: # Loop over episodes, i.e. over batches of scenarios
            selected, prob = self.model(state) # Compute solutions and probs using the NN-model
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected) # Performs transition and compute rewards
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2) # Update prob_list tensor

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2) # Note that none prod is performed, only sum
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # Filter the best results from pomo (from the trajectories)
        score_mean = -max_pomo_reward.float().mean()  # Multiplying for a negative sign to get positive values

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()