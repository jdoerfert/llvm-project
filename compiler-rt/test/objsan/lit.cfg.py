# -*- Python -*-

import os
import subprocess

# Setup config name.
config.name = "input-gen" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

config.suffixes = [".c", ".cpp"]
if config.host_os not in ["Linux"]:
    config.unsupported = True

def cxx_build_invocation(compile_flags, with_lto=False):
    lto_flags = []
    if with_lto and config.lto_supported:
        lto_flags += config.lto_flags

    return " " + " ".join([config.clang] + config.cxx_mode_flags + lto_flags + compile_flags) + " "

gen_args = '-mllvm --input-gen-mode=generate -flto -O2'.split(' ')
replay_gen_args = '-mllvm --input-gen-mode=replay_generated -flto -O2'.split(' ')

etc_args = []

gen_args_link = '-fuse-ld=lld -flto -linputgen.generate -lpthread -lstdc++ -O2'.split(' ') + etc_args
replay_gen_args_link = '-fuse-ld=lld -flto -linputgen.replay -lpthread -lstdc++ -O2'.split(' ') + etc_args

config.substitutions.append(
    ('%clangxx_inputgen_gen', cxx_build_invocation(gen_args)))
config.substitutions.append(
    ('%clangxx_inputgen_replay_gen', cxx_build_invocation(replay_gen_args)))

config.substitutions.append(
    ('%clangxx_inputgen_link_gen', cxx_build_invocation(gen_args_link)))
config.substitutions.append(
    ('%clangxx_inputgen_link_replay_gen', cxx_build_invocation(replay_gen_args_link)))
