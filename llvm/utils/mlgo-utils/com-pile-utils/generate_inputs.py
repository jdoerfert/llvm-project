#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""tool for making a corpus of loops from compile
"""

import argparse
import tempfile
import os
import json
import subprocess
from datasets import load_dataset
import dataclasses
import collections
import sys
from typing import Dict, Tuple, BinaryIO, Union, List, Optional, Iterable

from absl import logging

if not (sys.version_info.major == 3 and sys.version_info.minor >= 12):
    absl.error('This script needs python version >= 3.12')
    exit(1)

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Generating inputs for ComPileLoop'
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num', default=3, type=int)
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('-mclang', default=[], action='append')
    args = parser.parse_args()
    main(args)

def main(args):
    ds = load_dataset(args.dataset, split='train', streaming=True)
    l = []
    for i, data in enumerate(ds):
        with tempfile.TemporaryDirectory(dir=args.temp_dir, delete=(not args.save_temps)) as tmpdir:
            process_module(data, tmpdir, args.save_temps, args.mclang)
    print(collections.Counter(l))

def get_output(cmd, mod):
    logging.debug(f'cmd: {" ".join(cmd)}.')
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE) as proc:

        outs, errs = proc.communicate(
            input=mod)
        status = proc.wait()

        if status != 0:
            logging.error(f'Exit with status {status}')
            logging.error(f'cmd: {" ".join(cmd)}')
            logging.error(f'output:')
            logging.error(errs.decode('utf-8'))
            raise Exception(f'Command failed: {cmd}')

        return outs

def get_instrumented_module(mod, mode):
    cmd = f'opt -O3 --input-gen-mode={mode}'.split(' ')
    return get_output(cmd, mod)

def get_executable_for_mode(mod, mode, rt, flags):
    cmd = f'opt -O3 --input-gen-mode={mode}'.split(' ')
    instrumented_mod = get_output(cmd, mod)
    cmd = f'clang++ -x ir - -O3 {rt} -lpthread -flto -fuse-ld=lld -fno-exceptions -DNDEBUG -o -'.split(' ') + flags
    exe = get_output(cmd, instrumented_mod)
    return mod, exe

def process_module(data: Dict, working_dir: str, save_temps, flags: str):
    mod = data['module']

    _, gen_exec = get_executable_for_mode(mod, 'generate', '-linputgen.generate', flags)
    repl_mod, repl_exec = get_executable_for_mode(mod, 'replay_generated', '-linputgen.replay', flags)

    gen_exec_path = os.path.join(working_dir, 'gen')
    with open(gen_exec_path, 'wb') as f:
        f.write(gen_exec)

    cmd = [gen_exec_path]

    print(cmd)

    del data['module']
    data['repl_module'] = repl_mod

    return data

if __name__ == '__main__':
    parse_args_and_run()
