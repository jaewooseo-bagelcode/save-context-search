#!/bin/bash
# Build SCS and copy to plugin directory

set -e

echo "Building SCS..."
cargo build --release

echo "Copying binary to plugin/bin/..."
mkdir -p plugin/bin
cp target/release/scs plugin/bin/

echo "Done! Plugin ready at: $(pwd)/plugin"
