#!/usr/bin/env python

"""
Code to run behavior cloning given expert rollout data.
Example usage:
    python bc.py expert_data/Humanoid-v2.pkl experts/Humanoid-v2.pkl --iterations 100

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('model_config', type=str, help='Model configuration file')
    parser.add_argument('model_file', type=str, help='Model output file')
    parser.add_argument('model_dir', type=str, help='Model ouput dir')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train model')
    args = parser.parse_args()

    print('initializing empty policy based on expert policy architecture')
    model = load_policy.init_empty_policy(args.model_config)
    print('model initialized')

    with open(args.expert_data_file, 'rb') as f:
        data = pickle.loads(f.read())
    assert(sorted(data.keys()) == ['actions', 'observations'])
    x = data['observations']
    y = data['actions']

    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit(x, y, batch_size=32, epochs=args.epochs)

    model_file = os.path.join(args.model_dir, args.model_file)
    model.save(tmp_model_file)

if __name__ == '__main__':
    main()
