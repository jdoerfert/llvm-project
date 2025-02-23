#!/usr/bin/env python3

import argparse
import os
import json
import subprocess
from datasets import load_dataset
import collections

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Reading ComPileLoop'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num', default=3, type=int)
    parser.add_argument('--dump-llvm', default=False, action='store_true')
    args = parser.parse_args()
    main(args)

def main(args):
    ds = load_dataset(args.dataset, split='train', streaming=True)
    l = []
    for i, data in enumerate(ds):
        print(f'Processing loop {i}')
        process_module(data, l, args.dump_llvm)
    print(collections.Counter(l))

def process_module(data, l, dump_llvm):
    l.append(data['loop_trip_count'])
    if dump_llvm:
        bitcode_module = data['module']
        del data['module']
        dis_command_vector = ['llvm-dis', '-']
        with subprocess.Popen(
            dis_command_vector,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE) as dis_process:
            output = dis_process.communicate(
                input=bitcode_module)[0].decode('utf-8')
        print(data)
        print(output)
    else:
        del data['module']
        print(data)

if __name__ == '__main__':
    parse_args_and_run()
