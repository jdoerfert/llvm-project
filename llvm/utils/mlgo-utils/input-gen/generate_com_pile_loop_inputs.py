#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating ComPileLoop+Inputs from ComPileLoop
"""

import argparse
import tempfile
import re
import os
import json
import subprocess
from datasets import load_dataset
import dataclasses
import collections
import sys
import stat
import logging
import glob
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable

from input_gen_module import InputGenModule

logger = logging.getLogger(__name__)

if sys.version_info.major == 3 and sys.version_info.minor < 12:
    absl.error('This script needs python version >= 3.12')
    exit(1)

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Generating inputs for ComPileLoop'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('-mclang', default=[], action='append')
    parser.add_argument('-mllvm', default=[], action='append')
    parser.add_argument('-debug', default=False, action='store_true')
    args = parser.parse_args()
    main(args)

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ds = load_dataset(args.dataset, split='train', streaming=True)
    l = []
    for i, data in enumerate(ds):
        with tempfile.TemporaryDirectory(dir=args.temp_dir, delete=(not args.save_temps)) as tmpdir:
            process_module(data, tmpdir, args.save_temps, args.mclang, args.mllvm)

def process_module(data: Dict, working_dir: str, save_temps, mclang: List, mllvm: List):
    igm = InputGenModule(data['module'], working_dir, save_temps, mclang, mllvm, ['__llvm_extracted_loop'])
    if not igm.prepare():
        return
    assert igm.get_num_entries() == 1
    igm.generate()

    # TODO obtain inputs and package them
    # TODO we want to gather some info on the inputs such as size, est. runtime,
    # exit status, etc

if __name__ == '__main__':
    parse_args_and_run()
