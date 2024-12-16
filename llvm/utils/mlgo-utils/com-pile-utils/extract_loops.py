#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for making a corpus of loops from ComPile
"""

import argparse
import os
import tempfile
import subprocess
import json
import signal

import pandas
import pyarrow

from pyarrow import parquet
from datasets import load_dataset

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='A tool for making a LLVM IR loop dataset'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--language', default='c')
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--output-dataset', required=True)
    parser.add_argument('--num', default=3, type=int)
    args = parser.parse_args()
    LoopExtractor(args).main()

# 100MB
PARQUET_SIZE = 100 * 1000 * 1000
#PARQUET_SIZE = 10000

class LoopExtractor:
    def __init__(self, args):
        self.args = args

        self.dfs = []
        self.total_pfile_size = 0
        self.first_in_parquet = 0
        self.parquet_id = 0
        self.should_break = False
        self.i = 0

        signal.signal(signal.SIGUSR2, self.receive_should_break)
        signal.signal(signal.SIGUSR1, self.receive)

    def receive(self, signum, stack):
        print(f'Progress: module {self.i} size {self.total_pfile_size}')

    def receive_should_break(self, signum, stack):
        print(f'Will break')
        self.should_break = True

    def write_parquet(self):
        name = os.path.join(self.args.output_dataset, 'train-' + str(self.parquet_id) + '.parquet')
        if len(self.dfs) == 0:
            return
        print(f'Writing intermediate parquet {self.parquet_id} with estimated size {self.total_pfile_size} for modules {self.first_in_parquet} to {self.i}')
        df = pandas.concat(self.dfs)
        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        parquet.write_table(table, name, compression='NONE')

        self.dfs = []
        self.total_pfile_size = 0
        self.first_in_parquet = self.i
        self.parquet_id += 1

    def main(self):
        args = self.args
        ds = load_dataset(os.path.join(args.dataset, args.language), split='train', streaming=True)
        os.mkdir(args.output_dataset)

        i = 0

        for i, data in enumerate(ds):
            self.i = i
            module = data['content']
            language = data['language']
            new_df, size_estimate = process_module(module, language, i, args)
            self.total_pfile_size += size_estimate
            if new_df is not None:
                self.dfs.append(new_df)
            if self.total_pfile_size > PARQUET_SIZE:
                self.write_parquet()
            if i == args.num:
                print(f'Finished all {args.num}')
                break
            if self.should_break:
                print(f'Stopping at {i}')
                break
        print(f'Writing final parquet {i}')
        self.write_parquet()

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
