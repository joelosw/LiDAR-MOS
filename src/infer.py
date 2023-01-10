#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from model.modules.user import *
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')


if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        default=None,
        help='Which config to use. Use Config in model/ if no other specified'
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' +
                datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    parser.add_argument(
        '--monte-carlo', '-c',
        type=int, default=30,
        help='Number of samplings per scan'
    )


    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default=None,
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("Uncertainty", FLAGS.uncertainty)
    print("Monte Carlo Sampling", FLAGS.monte_carlo)
    print("infering", FLAGS.split)
    print("----------\n")
    #print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    if FLAGS.config is None:
        config_path = FLAGS.model + "/config.yml"
    else:
        config_path = FLAGS.config
    try:
        print("Opening config file from %s" % FLAGS.model)
        CONFIG = yaml.safe_load(open(config_path, 'r'))
    except Exception as e:
        print(e)
        print("Error opening config yaml file.")
        quit()
    sensor_name = CONFIG["dataset"]["sensor"]["name"]
    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in CONFIG["split"]["train"]:
            if seq is None: continue
            if not isinstance(seq, str):
                seq = '{0:02d}'.format(int(seq))
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, sensor_name, "predictions"))
        for seq in CONFIG["split"]["valid"]:
            if seq is None: continue
            if not isinstance(seq, str):
                seq = '{0:02d}'.format(int(seq))
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, sensor_name, "predictions"))
        for seq in CONFIG["split"]["test"]:
            if seq is None: continue
            if not isinstance(seq, str):
                seq = '{0:02d}'.format(int(seq))
            print("test", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, sensor_name, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user and infer dataset
    user = User(CONFIG, FLAGS.log, FLAGS.model,FLAGS.split,FLAGS.uncertainty,FLAGS.monte_carlo)
    user.infer()
