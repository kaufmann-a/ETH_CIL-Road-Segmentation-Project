"""
Main class for training
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import time
import argparse

from source.configuration import Configuration
from source.engine import Engine
from source.logcreator.logcreator import Logcreator
from source.helpers import argumenthelper


if __name__ == "__main__":
    global config
    #Sample Config: --handin true --configuration D:\GitHub\AML\Task1\configurations\test.jsonc
    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str,
                        help="Working directory (default: current directory).")
    parser.add_argument('--weights', default='', type=str, help="Optional pretrained weights of a model to continue training.")


    args = argumenthelper.parse_args(parser)
    start = time.time()

    Configuration.initialize(args.configuration, args.workingdir)
    Logcreator.initialize()

    Logcreator.h1("Some title")
    Logcreator.info("Environment: %s" % Configuration.get('environment.name'))

    engine = Engine()

    if args.weights:
        epoch, train_loss, train_acc, val_loss, val_acc = engine.load_checkpints(args.weights)
        engine.train(epoch)
    else:
        engine.train()

    end = time.time()
    Logcreator.info("Finished processing in %d [s]." % (end - start))
