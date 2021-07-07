"""
Main class for training
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import argparse
import os
import time

from source.configuration import Configuration
from source.engine import Engine
from source.helpers import argumenthelper
from source.logcreator.logcreator import Logcreator

if __name__ == "__main__":
    global config
    # Sample Config: --handin true --configuration D:\GitHub\AML\Task1\configurations\test.jsonc
    parser = argparse.ArgumentParser(description="Executes a training session.")
    parser.add_argument('--configuration', default='./configurations/default.jsonc',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str,
                        help="Working directory (default: current directory).")
    parser.add_argument('--weights', default='', type=str,
                        help="Optional pretrained weights of a model to continue training.")
    parser.add_argument('--resetlr',
                        default=False,
                        type=argumenthelper.boolean_string,
                        help="Optional: Reset the learning rate when the --weights parameter is set to load a checkpoint.")
    parser.add_argument('--lines_layer_path', default = '', type=str)
    parser.add_argument('--predicted_masks_path', default = '', type=str)

    args = argumenthelper.parse_args(parser)
    start = time.time()

    Configuration.initialize(args.configuration, args.workingdir, create_output_train=True, create_output_inf=False)
    Logcreator.initialize()

    Logcreator.h1("Some title")
    Logcreator.info("Environment: %s" % Configuration.get('environment.name'))

    engine = Engine(args)

    if args.weights:
        epoch, train_loss, train_acc, val_loss, val_acc = engine.load_checkpoints(args.weights, args.resetlr)
        engine.train(epoch)
    else:
        engine.train()

    end = time.time()
    Logcreator.info("Finished processing in %d [s]." % (end - start))
