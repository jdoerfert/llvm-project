#!/usr/bin/env python3

import argparse
import json
import signal
import pandas
import pyarrow
import os

from pyarrow import parquet
from datasets import load_dataset

# 100MB
PARQUET_SIZE = 100 * 1000 * 1000

class DatasetWriter:
    def __init__(self, begin, end, parquet_start, output_dataset, output_dataset_json, parquet_size=PARQUET_SIZE):
        self.output_dataset = output_dataset
        self.output_dataset_json = output_dataset_json
        self.parquet_size = parquet_size

        self.begin = begin
        self.end = end

        self.dfs = []
        self.total_pfile_size = 0
        self.parquet_id = parquet_start
        self.should_break = False
        self.i = begin
        self.first_in_parquet = self.i

        signal.signal(signal.SIGUSR2, self.receive_should_break)
        signal.signal(signal.SIGUSR1, self.receive)

    def receive(self, signum, stack):
        print(f'Progress: module {self.i} size {self.total_pfile_size}')

    def receive_should_break(self, signum, stack):
        print(f'Will break')
        self.should_break = True

    def get_current_parquet_name(self):
        return os.path.join(self.output_dataset, 'train-' + str(self.parquet_id) + '.parquet')

    def get_current_parquet_json_name(self):
        return os.path.join(self.output_dataset_json, 'train-' + str(self.parquet_id) + '.parquet.json')

    def write_parquet(self):
        name = self.get_current_parquet_name()
        json_name = self.get_current_parquet_json_name()
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

    def process(self, process_fn, ds):
        curparname = self.get_current_parquet_name()
        if os.path.exists(curparname):
            raise Exception(f'The parquet file {curparname} already exists. Aborting.')
        curjsonname = self.get_current_parquet_json_name()
        if os.path.exists(curjsonname):
            raise Exception(f'The parquet json file {curjsonname} already exists. Aborting.')

        os.makedirs(self.output_dataset, exist_ok=True)
        os.makedirs(self.output_dataset_json, exist_ok=True)

        ds = ds.skip(self.i)

        for data in ds:
            new_df, size_estimate = process_fn(self.i, data)
            self.total_pfile_size += size_estimate
            if new_df is not None:
                self.dfs.append(new_df)
            if self.total_pfile_size > PARQUET_SIZE:
                self.write_parquet()
            if self.i == self.end:
                print(f'Finished all {self.end}')
                break
            if self.should_break:
                print(f'Stopping at {self.i}')
                break
            self.i += 1

        print(f'Writing final parquet {self.i}')
        self.write_parquet()
