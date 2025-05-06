#!/usr/bin/bash
set -eux pipefail
time tar -cf constraints-as-terminations-logs-and-runs.tar.zst -I 'zstd --ultra -22 -T0' ./logs/clean_rl/

# To install: wget -qO- https://github.com/Blutsh/Swish/releases/download/1.0.1/swish-1.0.1-x86_64-unknown-linux-musl.tar.gz | tar xz --strip-components=1 -C ~/.local/bin/ --wildcards '*/swish'
swish constraints-as-terminations-logs-and-runs.tar.zst
