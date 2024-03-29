#!/usr/bin/env python3
# coding: utf8

"""
Adds environment variables from .env file.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import sys

"""
Title: python-dotenv
Author: Saurabh Kumar
Code version: 0.10.1
Availability: https://github.com/theskumar/python-dotenv
"""


def _find_dotenv(filename='.env', raise_error_if_not_found=False, usecwd=False):
    """
    Search in increasingly higher folders for the given file

    Returns path to the file if found, or an empty string otherwise
    """
    if usecwd or '__file__' not in globals():
        # should work without __file__, e.g. in REPL or IPython notebook
        path = os.getcwd()
    else:
        # will work for .py files
        frame = sys._getframe()
        # find first frame that is outside of this file
        while frame.f_code.co_filename == __file__:
            frame = frame.f_back
        frame_filename = frame.f_code.co_filename
        path = os.path.dirname(os.path.abspath(frame_filename))

    for dirname in _walk_to_root(path):
        check_path = os.path.join(dirname, filename)
        if os.path.isfile(check_path):
            return check_path

    if raise_error_if_not_found:
        raise IOError('File not found')

    return ''


def _walk_to_root(path):
    """
    Yield directories starting from the given directory up to the root
    """
    if not os.path.exists(path):
        raise IOError('Starting path not found')

    if os.path.isfile(path):
        path = os.path.dirname(path)

    last_dir = None
    current_dir = os.path.abspath(path)
    while last_dir != current_dir:
        yield current_dir
        parent_dir = os.path.abspath(os.path.join(current_dir, os.path.pardir))
        last_dir, current_dir = current_dir, parent_dir


def initialize(extend=False):
    """
    Load the current dotenv as system environemt variable.
    """
    try:
        with open(_find_dotenv(), 'r') as fp:
            for line in fp:
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                shouldextend = False
                if k.endswith("+"):
                    shouldextend = True
                    k = k.rstrip("+")

                v = v.rstrip('\n')
                if k in os.environ and not extend:
                    continue
                elif k in os.environ and extend and shouldextend:
                    os.environ[k] += os.pathsep + v
                else:
                    os.environ[k] = v
        return True
    except:
        return False
