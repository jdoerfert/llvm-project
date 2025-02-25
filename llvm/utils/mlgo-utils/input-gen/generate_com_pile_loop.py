#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating ComPileLoop from ComPile
"""

import argparse
import os
import tempfile
import subprocess
import json
import signal
import sys
import pandas

from datasets import load_dataset
from dataset_writer import DatasetWriter

if sys.version_info.major == 3 and sys.version_info.minor < 12:
    absl.error('This script needs python version >= 3.12')
    exit(1)

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='A tool for making a LLVM IR loop dataset'
    )
    parser.add_argument('--language', default='c')
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('--temp-dir', default=None)

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    parser.add_argument('--output-dataset-json', required=True)
    parser.add_argument('--begin', default=0, type=int)
    parser.add_argument('--end', default=3, type=int)
    parser.add_argument('--parquet-start', default=0, type=int)

    args = parser.parse_args()

    ds = load_dataset(os.path.join(args.dataset, args.language), split='train', streaming=True)

    dw = DatasetWriter(args.begin, args.end, args.parquet_start, args.output_dataset, args.output_dataset_json)
    le = LoopExtractor(args)
    dw.process(le.process_module_wrapper, ds)

class LoopExtractor:
    def __init__(self, args):
        self.args = args

    def process_module_wrapper(self, i, data):
        return process_module(data['content'], data['language'], i, self.args)

def process_module(module, language, idx, args):
    with tempfile.TemporaryDirectory(dir=args.temp_dir, delete=(not args.save_temps)) as outdir:
        return process_module_in_dir(module, language, idx, outdir)

def process_module_in_dir(module, language, idx, temp_outdir):
    size_estimate = 0

    prefix = str(os.path.join(temp_outdir, 'output.'))
    suffix = '.bc'
    cmd = [
        'llvm-extract-loops',
        '-',
        '--output-prefix', prefix,
        '--output-suffix', suffix,
    ]
    verbose = False
    if verbose:
        print(' '.join(cmd))
    with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE) as proc:
        output = proc.communicate(
            input=module)[0].decode('utf-8')

    dfs = []
    i = 0
    while True:
        try:
            module_path = prefix + str(i) + suffix
            metadata_path = module_path + '.json'

            module_file = open(module_path, 'br')
            loop_module = module_file.read()
            module_file.close()

            metadata_file = open(metadata_path, 'r')
            data = json.load(metadata_file)
            metadata_file.close()

            data['language_in_compile'] = language
            data['module_idx_in_compile'] = idx
            data['module'] = loop_module

            size_estimate += len(loop_module)

            dfs.append(pandas.DataFrame(data, index=[0]))

        except OSError as e:
            break
        i += 1

    if len(dfs) == 0:
        return None, 0

    return pandas.concat(dfs), size_estimate

if __name__ == '__main__':
    parse_args_and_run()
