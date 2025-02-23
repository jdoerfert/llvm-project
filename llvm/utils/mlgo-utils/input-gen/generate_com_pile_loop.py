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
import pyarrow

from pyarrow import parquet
from datasets import load_dataset

if sys.version_info.major == 3 and sys.version_info.minor < 12:
    absl.error('This script needs python version >= 3.12')
    exit(1)

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='A tool for making a LLVM IR loop dataset'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--language', default='c')
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--output-dataset', required=True)
    parser.add_argument('--begin', default=0, type=int)
    parser.add_argument('--end', default=3, type=int)
    parser.add_argument('--parquet-start', default=0, type=int)
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
        self.parquet_id = args.parquet_start
        self.should_break = False
        self.i = args.begin
        self.first_in_parquet = self.i

        signal.signal(signal.SIGUSR2, self.receive_should_break)
        signal.signal(signal.SIGUSR1, self.receive)

    def receive(self, signum, stack):
        print(f'Progress: module {self.i} size {self.total_pfile_size}')

    def receive_should_break(self, signum, stack):
        print(f'Will break')
        self.should_break = True

    def get_current_parquet_name(self):
        return os.path.join(self.args.output_dataset, 'train-' + str(self.parquet_id) + '.parquet')

    def write_parquet(self):
        name = self.get_current_parquet_name()
        json_name = name + '.json'
        if len(self.dfs) == 0:
            return
        print(f'Writing intermediate parquet {self.parquet_id} with estimated size {self.total_pfile_size} for modules {self.first_in_parquet} to {self.i}')
        df = pandas.concat(self.dfs)
        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        parquet.write_table(table, name, compression='NONE')
        with open(json_name, 'w') as fp:
            fp.write(json.dumps({
                'estimated_size' : self.total_pfile_size,
                'first' : self.first_in_parquet,
                'last' : self.i,
                'num' : self.i - self.first_in_parquet + 1,
            }, indent=4) + '\n')

        self.dfs = []
        self.total_pfile_size = 0
        self.first_in_parquet = self.i + 1
        self.parquet_id += 1

    def main(self):
        args = self.args
        ds = load_dataset(os.path.join(args.dataset, args.language), split='train', streaming=True)
        os.makedirs(args.output_dataset, exist_ok=True)

        curparname = self.get_current_parquet_name()
        if os.path.exists(curparname):
            print(f'The parquet name {curparname} already exists. Aborting.')
            return

        data = ds.skip(self.i)

        for data in ds:
            module = data['content']
            language = data['language']
            new_df, size_estimate = process_module(module, language, self.i, args)
            self.total_pfile_size += size_estimate
            if new_df is not None:
                self.dfs.append(new_df)
            if self.total_pfile_size > PARQUET_SIZE:
                self.write_parquet()
            if self.i == args.end:
                print(f'Finished all {args.end}')
                break
            if self.should_break:
                print(f'Stopping at {self.i}')
                break
            self.i += 1

        print(f'Writing final parquet {self.i}')
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
