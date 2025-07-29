##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = False
CUDA_DEVICE_NUM = 0

##########################################################################################
# Importing modules

import logging
import os
import sys
#from CVRPTrainer import CVRPTrainer as Trainer
#from utils.utils import create_logger, copy_all_src

##########################################################################################
# Setting Parameters

env_params = {
    'problem_size': 30,
    'pomo_size': 30,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [8001, 8051],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 5,
    'train_episodes': 128, # Total number of generated scenarios for the problem, for training we have used 1280000
    'train_batch_size': 64,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 1,
    },
    'model_load': {
        'enable': False,  # This enables to load a pre-trained model
        'path': '/content',  # Directory path of the pre-trained model
        'epoch': 10,  # Truncated epoch of the pre-trained model to load
    }
}

logger_params = {
    'log_file': {
        'desc': '',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = CVRPTrainer(env_params=env_params,
                          model_params=model_params,
                          optimizer_params=optimizer_params,
                          trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

def _set_debug_mode(): # To run a small test of the functionality of the code
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
  main()