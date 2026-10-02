default:
  just --list --list-submodules

check: check-format check-clippy check-test check-doc check-features

check-format:
  cargo fmt --all -- --check

check-clippy:
  cargo clippy --workspace --all-targets -- -D warnings

check-test:
  cargo test

check-doc:
  RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

check-features:
  RUSTFLAGS="-D warnings" cargo check --workspace --all-targets
  RUSTFLAGS="-D warnings" cargo hack check \
    --feature-powerset \
    --no-dev-deps

train-and-run prompt="ROMEO:":
  cargo run --release -- train --preset tiny --input data/input.txt | \
    cargo run --release -- generate --num-tokens 500 {{prompt}}

flamegraph preset="tiny" iterations="100" input="data/input.txt" output="./out/model-flamegraph.bin":
  mkdir -p out
  CARGO_PROFILE_RELEASE_DEBUG=true \
    CARGO_PROFILE_RELEASE_STRIP=none \
    RUSTFLAGS="-C link-arg=-Wl,--no-rosegment" \
    cargo flamegraph --release \
      -o out/flamegraph.svg \
      -c "record -F 997 --call-graph dwarf,64000 -g -o ./out/perf.data" \
      -- train --preset {{preset}} \
      --iterations {{iterations}} \
      --input {{input}} \
      --output {{output}}
  wc -c < {{output}} | numfmt --to=iec-i --suffix=B

