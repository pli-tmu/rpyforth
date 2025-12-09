#!/bin/bash

benchmarks=(fibo.fs ack.fs nestedloop.fs sieve.fs heap.fs ary.fs)
commands=(gforth ./rpyforth.sh)

if [ ! -d logs ]; then
    mkdir logs
fi

for bm in "${benchmarks[@]}"; do
    for cmd in "${commands[@]}"; do
        cmd_cleaned="${cmd#./}"
        echo "Running ${cmd_cleaned} against ${bm}..."
        for i in `seq 1 100`; do
            ${cmd} shootout/${bm} > logs/${bm}_${cmd_cleaned}_${i}.log
        done
    done
done
