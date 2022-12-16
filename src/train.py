#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import os
import shutil
from shutil import copyfile
import model.__init__ as booger
import yaml
from model.modules.trainer import *
from pip._vendor.distlib.compat import raw_input

from model.modules.SalsaNextAdf import *
from model.modules.SalsaNext import *
#from src.model.modules.save_dataset_projected import *
import math
from decimal import Decimal

def remove_exponent(d):
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

def millify(n, precision=0, drop_nulls=True, prefixes=[]):
    millnames = ['', 'k', 'M', 'B', 'T', 'P', 'E', 'Z', 'Y']
    if prefixes:
        millnames = ['']
        millnames.extend(prefixes)
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    result = '{:.{precision}f}'.format(n / 10**(3 * millidx), precision=precision)
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    return '{0}{dx}'.format(result, dx=millnames[millidx])


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
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='config/master_config.yml',
        help='Main Config file, where everything else is included',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default="~/output",
        help='Directory to put the log data. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default="",
        help='If you want to give an aditional discriptive name'
    )
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=False,
        default="",
        help='Directory to get the pretrained model. If not passed, do from scratch!'
    )
    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=True,
        help='Set this if you want to use the Uncertainty Version'
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.log = FLAGS.log + '/logs/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + FLAGS.name
    
    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CONFIG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    
    if FLAGS.uncertainty:
        model = SalsaNextUncertainty(20)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model = SalsaNext(nclasses = 20, num_last_n = len(CONFIG['residual_images']['num_last_n']))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("config", FLAGS.config)
    print("uncertainty", FLAGS.uncertainty)
    print("Total of Trainable Parameters: {}".format(millify(pytorch_total_params,2)))
    print("log: ", FLAGS.log)
    print("pretrained: ", FLAGS.pretrained)
    print("----------\n")
    # print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # create log folder
    try:
        if FLAGS.pretrained == "":
            FLAGS.pretrained = None
            if os.path.isdir(FLAGS.log):
                if os.listdir(FLAGS.log):
                    answer = raw_input("Log Directory is not empty. Do you want to proceed? [y/n]  ")
                    if answer == 'n':
                        quit()
                    else:
                        shutil.rmtree(FLAGS.log)
            os.makedirs(FLAGS.log)
        else:
            FLAGS.log = FLAGS.pretrained
            print("Not creating new log file. Using pretrained directory")
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if FLAGS.pretrained is not None:
        if os.path.isdir(FLAGS.pretrained):
            print("model folder exists! Using model from %s" % (FLAGS.pretrained))
        else:
            print("model folder doesn't exist! Start with random weights...")
    else:
        print("No pretrained directory found.")

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("Dumping Config to %s for further reference." % FLAGS.log)
        with open(os.path.join(FLAGS.log, "config.yml"), 'w') as file:
            documents = yaml.dump(CONFIG, file)
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

    # create trainer and start the training
    trainer = Trainer(master_config = CONFIG, logdir = FLAGS.log, pretrained_path = FLAGS.pretrained, uncertainty = FLAGS.uncertainty)
    trainer.train()
