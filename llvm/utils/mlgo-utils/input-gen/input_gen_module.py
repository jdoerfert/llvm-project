#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for generating inputs for an llvm module
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

logger = logging.getLogger(__name__)

if sys.version_info.major == 3 and sys.version_info.minor < 12:
    absl.error('This script needs python version >= 3.12')
    exit(1)

class InputGenModule:
    def __init__(self, mod, working_dir, save_temps, mclang, mllvm, entries='marked'):
        self.mclang = mclang
        self.mllvm = mllvm
        self.save_temps = save_temps
        self.mod = mod
        self.working_dir = working_dir
        self.entries = entries

        self.num_entries = None

        self.gen_mod, self.gen_exec = None, None
        self.repl_mod, self.repl_exec = None, None

        self.save_temps_counter = 0

    def save_temp(self, content, name='temp', binary=True):
        if not self.save_temps:
            return

        self.save_temps_counter += 1
        if binary:
            mode = 'wb'
        else:
            mode = 'w'
        fn = os.path.join(self.working_dir, str(self.save_temps_counter) + '_intermediate_' + name)
        logger.info(f'Saving temp {fn}')
        with open(fn, mode) as f:
            f.write(content)

    def get_output(self, cmd, stdin=None, allow_fail=False):
        logger.debug(f'Running cmd: {" ".join(cmd)}.')
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=(subprocess.PIPE if stdin is not None else None)) as proc:

            outs, errs = proc.communicate(
                input=stdin)
            status = proc.wait()

            if status != 0 and not allow_fail:
                logger.debug(f'Exit with status {status}')
                logger.debug(f'cmd: {" ".join(cmd)}')
                logger.debug(f'output:')
                logger.debug(errs.decode('utf-8'))
                raise Exception(f'Command failed: {cmd}')

            return outs, errs

    def get_instrumented_module(self, mod, mode):
        cmd = f'opt -O3 --input-gen-mode={mode}'.split(' ') + self.mllvm

        if self.entries == 'all':
            cmd.append('--input-gen-entry-all-functions')
        elif self.entries == 'marked':
            pass
        else:
            for func_name in self.entries:
                cmd.append('--input-gen-entry-function=' + func_name)

        return self.get_output(cmd, mod)

    def get_executable_for_mode(self, mod, mode, rt):
        instrumented_mod, _ = self.get_instrumented_module(mod, mode)
        cmd = f'clang++ -x ir - -O3 {rt} -lpthread -flto -fuse-ld=lld -fno-exceptions -DNDEBUG -o -'.split(' ') + self.mclang
        exe, _ = self.get_output(cmd, instrumented_mod)
        self.save_temp(exe, mode + '.exe', binary=True);
        self.save_temp(exe, mode + 'instrumented_mod', binary=True);
        return instrumented_mod, exe

    def prepare(self):
        self.save_temp(self.mod, 'original_module', binary=True);

        self.gen_mod, self.gen_exec = self.get_executable_for_mode(self.mod, 'generate', '-linputgen.generate')
        self.repl_mod, self.repl_exec = self.get_executable_for_mode(self.mod, 'replay_generated', '-linputgen.replay')

        # Write the generation executable to file
        self.gen_exec_path = os.path.join(self.working_dir, 'gen')
        with open(self.gen_exec_path, 'wb') as f:
            f.write(self.gen_exec)

        # Make executable by user
        st = os.stat(self.gen_exec_path)
        os.chmod(self.gen_exec_path, st.st_mode | stat.S_IXUSR)

        cmd = [self.gen_exec_path, '-1']
        _, errs = self.get_output(cmd, allow_fail=True)
        re_match = re.search('  Num available functions: ([0-9]+)', errs.decode('utf-8'))

        if re_match is None:
            return False

        self.num_entries = int(re_match.group(1))

        return True

    def generate(self, entry_no=0, num_inputs=1, num_threads=1, first_input=0, seed=42):
        cmd = [
            self.gen_exec_path,
            str(entry_no),
            str(num_inputs),
            str(num_threads),
            str(first_input),
            str(seed)
        ]
        outs, errs = self.get_output(cmd)

        logger.debug(f'Outs: {outs.decode("utf-8")}')
        logger.debug(f'Errs: {errs.decode("utf-8")}')

        inputs = glob.glob(self.gen_exec_path + '*.inp')
        logger.debug(f'Inputs: {inputs}')

    def get_num_entries(self):
        return self.num_entries

    def get_repl_mod(self):
        return self.repl_mod

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        description='Generating inputs for a module'
    )
    parser.add_argument('--module', required=True)
    parser.add_argument('--temp-dir', default=None)
    parser.add_argument('--save-temps', action='store_true', default=False)
    parser.add_argument('-mclang', default=[], action='append')
    parser.add_argument('-mllvm', default=[], action='append')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('--entry-function', default=[], action='append')
    parser.add_argument('--entry-all', default=False, action='store_true')
    parser.add_argument('--entry-marked', default=False, action='store_true')
    args = parser.parse_args()
    main(args)

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if int(args.entry_marked) + int(args.entry_all) + int(len(args.entry_function) > 0) != 1:
        logger.error('Exactly one of `--entry-function`, `--entry-all`, or `--entry-marked` must be specified')
        return

    mode = 'invalid'
    if args.entry_marked:
        mode = 'marked'
    elif args.entry_all:
        mode = 'all'
    else:
        mode = args.entry_function

    with open(args.module, 'rb') as f:
        mod = f.read()

    with tempfile.TemporaryDirectory(dir=args.temp_dir, delete=(not args.save_temps)) as tmpdir:
        igm = InputGenModule(mod, tmpdir, args.save_temps, args.mclang, args.mllvm, mode)
        if not igm.prepare():
            logger.error('Module preparation failed')
            return
        igm.generate()

if __name__ == '__main__':
    parse_args_and_run()
