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

etc_args = []

inputgen_generate_lib_path = os.path.join(
            config.compiler_rt_libdir,
            "libinputgen.generate.a")
inputgen_replay_lib_path = os.path.join(
            config.compiler_rt_libdir,
            "libinputgen.replay.a")

gen_args = '-g -mllvm --input-gen-allow-external-funcs=printf -mllvm --input-gen-mode=generate -flto -O2'.split(' ') + etc_args
replay_gen_args = '-g -mllvm --input-gen-allow-external-funcs=printf -mllvm --input-gen-mode=replay_generated -flto -O2'.split(' ') + etc_args

gen_args_link = f'-g -fuse-ld=lld -flto {inputgen_generate_lib_path} -lpthread -lstdc++ -O2'.split(' ') + etc_args
replay_gen_args_link = f'-g -fuse-ld=lld -flto {inputgen_replay_lib_path} -lpthread -lstdc++ -O2'.split(' ') + etc_args

gen_args_full = 'rm -rf %t.gen.exe.*.inp && ' + cxx_build_invocation(gen_args) + ' -c %s -o %t.gen.o && ' + cxx_build_invocation(gen_args_link) + ' %t.gen.o -o %t.gen.exe'
replay_gen_args_full = cxx_build_invocation(replay_gen_args) + ' -c %s -o %t.replay_gen.o && ' + cxx_build_invocation(replay_gen_args_link) + ' %t.replay_gen.o -o %t.replay_gen.exe'

config.substitutions.append(('%inputgen_gen', '%t.gen.exe'))
config.substitutions.append(('%inputgen_repl_gen', '%t.replay_gen.exe'))

config.substitutions.append(
    ('%clangxx_inputgen_gen', cxx_build_invocation(gen_args)))
config.substitutions.append(
    ('%clangxx_inputgen_replay_gen', cxx_build_invocation(replay_gen_args)))

config.substitutions.append(
    ('%clangxx_inputgen_link_gen', cxx_build_invocation(gen_args_link)))
config.substitutions.append(
    ('%clangxx_inputgen_link_replay_gen', cxx_build_invocation(replay_gen_args_link)))

replay_gen_args = '-mllvm --input-gen-mode=replay_generated -flto -O2'.split(' ')

config.substitutions.append(
    ('%clangxx_inputgen_full_gen', gen_args_full))

config.substitutions.append(
    ('%clangxx_inputgen_full_replay_gen', replay_gen_args_full))
