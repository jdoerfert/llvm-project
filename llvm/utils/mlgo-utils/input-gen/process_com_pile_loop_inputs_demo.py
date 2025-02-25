#!/usr/bin/env python3

import argparse
import os
import json
import logging
import subprocess
import dataclasses
from datasets import load_dataset
import collections

from input_gen_module import InputGenReplay, Input

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Reading ComPileLoop'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num', default=3, type=int)
    parser.add_argument('--dump-llvm', default=False, action='store_true')

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
        print(f'Processing loop {i}')
        process_module(data, l, args.dump_llvm, args)
    print(collections.Counter(l))

def process_module(data, l, dump_llvm, args):
    l.append(data['loop_trip_count'])
    igm = InputGenReplay(
        data['module'],
        working_dir=None,
        save_temps=args.save_temps,
        mclang=args.mclang,
        mllvm=args.mllvm,
        temp_dir=args.temp_dir,
    )

    igm.prepare()
    for inpt in data['inputs']:
        inpt = Input(**inpt)
        num = 2
        print(f'Replayed {num} times')
        print(list(igm.replay_input(inpt.data, inpt.entry_no, num)))

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
