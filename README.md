# halo2curves MSM Acceleration

This package provides GPU-accelerated Multi-Scalar Multiplication (MSM) for the `halo2curves` library, optimized for Apple M-series chips and iPhones. It supports both `halo2curves` and `arkworks` curve implementations, with a primary focus on the BN254 curve.

## Features

1. **Curve Support**:
   - `halo2curves` (use the `h2c` feature)
   - `arkworks` (use the `ark` feature)
   - Only BN254 is supported at the moment.

2. **Platform-Specific Shader Compilation**:
   - For macOS, use the `macos` feature.
   - For iOS, use the `ios` feature.
   - **Note**: These features are mutually exclusive as Metal shaders are compiled during the build process (`build.rs`).

## Testing

To run all tests:
```
cargo test --features "h2c ark macos" --no-default-features --release
```

Some `arkworks` MSM tests are ignored by default due to concurrency issues. Run these separately with:
```
cargo test --features "h2c ark macos" --no-default-features --release -- --test-threads=1 --ignored
```

## Benchmarking

1. **General Benchmarking**:
   ```
   cargo bench -- benchmark_msm
   ```
   This benchmarks:
   - The `halo2curves` implementation
   - The GPU-accelerated implementation
   - For log size 20 and 5 instances by default.
   
    To benchmark the `arkworks` implementation, use the `ark` feature:
   ```
    cargo bench --features "ark" -- benchmark_msm
    ```
2. **Custom Benchmarking**:
   Use the `gpu_profiler` binary for fine-grained control:
   ```
   RUST_LOG=info cargo run --release --bin gpu_profiler 20 5 gpu_cpu 10
   ```
   - **Arguments**:
     - First: MSM log size
     - Second: Number of instances
     - Third: Algorithm to use (refer to `gpu_profiler` source for options)
     - Fourth: Number of reruns for consistent results.
     - **Note**: You can also change the logging level to `debug` for more detailed output on the GPU implementation.

3. **iOS Benchmarking**:
   The `ios-metal-benchmarker` project allows performance measurement on iPhones:
   - Generate iOS bindings:
     ```
     cargo run --release --bin gen_ios_bindings
     ```
   - Follow the Mopro iOS setup guide for integrating the bindings:
     [iOS Setup Tutorial](https://zkmopro.org/docs/setup/ios-setup)

## Installation

Ensure you have a macOS machine with an M-series chip (or an iPhone for iOS benchmarks). Compilation for iOS still requires a macOS environment.

## Contributions

Contributions are welcome! Please open issues for feature requests or bug reports.

## License

This project is licensed under the MIT License. See the LICENSE file for details.