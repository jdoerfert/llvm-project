#!/bin/sh
set -x
set -e

basen=${1%.*}
reco=$basen.rec.o
replo=$basen.repl.o

clang -O3 -flto -c $1 -o $reco -mllvm --input-gen-mode=generate
clang -O3 -flto -c $1 -o $replo -mllvm --input-gen-mode=replay
clang++ ig.cpp vm_obj.cpp vm_storage.cpp recorder.cpp -g -std=c++20 -O3 $reco -o $basen.rec -flto -fuse-ld=lld -fno-exceptions -DNDEBUG
clang++ ig.cpp vm_obj.cpp vm_storage.cpp recorder.cpp -g -std=c++20 -O3 $reco -o $basen.rec.dbg -flto -fuse-ld=lld -fno-exceptions
clang++ vm_storage.cpp replay.cpp  -g -std=c++20 -O3 $replo -o $basen.repl -flto -fuse-ld=lld -fno-exceptions -DNDEBUG
clang++ vm_storage.cpp replay.cpp  -g -std=c++20 -O3 $replo -o $basen.repl.dbg -flto -fuse-ld=lld -fno-exceptions
