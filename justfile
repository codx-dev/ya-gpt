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

train-naive preset="small" iterations="100" input="data/input.txt" output="./out/model-naive.bin":
  RUSTFLAGS="-C target-cpu=native" \
    cargo run --release -p ya-gpt-cli -- train \
      --engine naive \
      --preset {{preset}} \
      --iterations {{iterations}} \
      --input {{input}} \
      --output {{output}}

train-simd-wide preset="small" iterations="100" input="data/input.txt" output="./out/model-simd-wide.bin":
  RUSTFLAGS="-C target-cpu=native" \
    cargo run --release -p ya-gpt-cli --features simd-wide -- train \
      --engine simd-wide \
      --preset {{preset}} \
      --iterations {{iterations}} \
      --input {{input}} \
      --output {{output}}

generate-naive tokens="500" input="./out/model-naive.bin" prompt="ROMEO:":
  RUSTFLAGS="-C target-cpu=native" \
    cargo run --release -p ya-gpt-cli  -- generate \
      --engine naive \
      --input {{input}} \
      --num-tokens {{tokens}} \
      {{prompt}}

generate-simd-wide tokens="500" input="./out/model-simd-wide.bin" prompt="ROMEO:":
  RUSTFLAGS="-C target-cpu=native" \
    cargo run --release -p ya-gpt-cli --features simd-wide -- generate \
      --engine simd-wide \
      --input {{input}} \
      --num-tokens {{tokens}} \
      {{prompt}}

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

