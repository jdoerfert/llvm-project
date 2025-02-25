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
import dataclasses
import collections
import sys
import stat
import logging
import glob
import pandas

from datasets import load_dataset
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable

from input_gen_module import InputGenModule, Input
from dataset_writer import DatasetWriter

logger = logging.getLogger(__name__)

if sys.version_info.major == 3 and sys.version_info.minor < 12:
    absl.error('This script needs python version >= 3.12')
    exit(1)

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Generating inputs for ComPileLoop'
    )
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('-mclang', default=[], action='append')
    parser.add_argument('-mllvm', default=[], action='append')
    parser.add_argument('-debug', default=False, action='store_true')

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    parser.add_argument('--output-dataset-json', required=True)
    parser.add_argument('--begin', default=0, type=int)
    parser.add_argument('--end', default=3, type=int)
    parser.add_argument('--parquet-start', default=0, type=int)

    args = parser.parse_args()
    main(args)

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ds = load_dataset(args.dataset, split='train', streaming=True)
    dw = DatasetWriter(args.begin, args.end, args.parquet_start, args.output_dataset, args.output_dataset_json)
    le = ComPileLoopInput(args)
    dw.process(le.process_module_wrapper, ds)

class ComPileLoopInput:
    def __init__(self, args):
        self.args = args

    def process_module_wrapper(self, i, data):
        try:
            igm = InputGenModule(
                data['module'],
                working_dir=None,
                save_temps=self.args.save_temps,
                mclang=self.args.mclang,
                mllvm=self.args.mllvm,
                entries=['__llvm_extracted_loop'])

            igm.prepare()
            assert igm.get_num_entries() == 1
            igm.generate(entry_no=0, num_inputs=5)

            data['inputs'] = None
            data['module'] = igm.get_repl_mod()
            size = len(data['module'])
            inputs = [dataclasses.asdict(i) for i in igm.get_generated_inputs()]

            logger.debug(data['module'])
            logger.debug(inputs)

            df = pandas.DataFrame(data, index=[0])
            df.at[0, 'inputs'] = inputs

            # TODO we want to gather some info on the inputs such as size, est. runtime,
            # We should also probably run the generated inputs and make sure they
            # run successfully.

            return df, size

        except Exception as e:
            logging.debug(f'InputGenModule failed: {e}')
            return None, 0

if __name__ == '__main__':
    parse_args_and_run()
