//
//  ContentView.swift
//  gpu-acceleration
//
//  Created by Artem Grigor on 08/01/2025.
//

import SwiftUI

struct ContentView: View {
    // The user-adjustable log size, initialized to 18 (valid range >= 10).
    @State private var logSize: Int = 18

    // Last single-run results for GPU+CPU and CPU-only.
    @State private var lastMetalCPUResult: UInt64 = 0
    @State private var lastCPUOnlyResult: UInt64 = 0

    // Averages for 100-run benchmark.
    @State private var avgMetalCPUResult: Double = 0
    @State private var avgCPUOnlyResult: Double = 0

    // UI/Thread control
    @State private var isRunning = false        // Blocks buttons & shows progress
    @State private var elapsedSeconds = 0       // Count how long a run takes
    @State private var timer: Timer?            // Increment `elapsedSeconds`

    var body: some View {
        VStack(spacing: 20) {
            Text("MSM Benchmark")
                .font(.headline)
                .padding(.top, 20)

            // Stepper controlling logSize (minimum 10)
            Stepper("Log Size: \(logSize)", value: $logSize, in: 10...100)
                .padding(.horizontal)
                .disabled(isRunning)

            // GPU + CPU Button
            Button("Run GPU + CPU (Metal)") {
                runSingleBenchmark(isMetal: true)
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)

            Text("Last GPU+CPU Run: \(lastMetalCPUResult) ms")

            // CPU-Only Button
            Button("Run CPU Only") {
                runSingleBenchmark(isMetal: false)
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)

            Text("Last CPU-Only Run: \(lastCPUOnlyResult) ms")

            // 100-iteration Benchmark
            Button("Run Long Benchmark (100 Iterations)") {
                runHundredIterations()
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)

            Text("Avg GPU+CPU (100 runs): \(String(format: "%.2f", avgMetalCPUResult)) ms")
            Text("Avg CPU-Only (100 runs): \(String(format: "%.2f", avgCPUOnlyResult)) ms")

            // ProgressView + elapsed time display
            if isRunning {
                ProgressView("Running... \(elapsedSeconds)s elapsed")
                    .padding()
            }

            Spacer()
        }
        .padding()
    }

    // MARK: - Single Run (GPU+CPU or CPU Only)

    private func runSingleBenchmark(isMetal: Bool) {
        // Block UI & start timer
        isRunning = true
        startTimer()

        // Run in background
        DispatchQueue.global(qos: .userInitiated).async {
            let ms: UInt64
            let size = UInt32(logSize)

            // Call the appropriate Rust function
            if isMetal {
                ms = benchmarkH2cMetalAndCpuMsmBest(logSize: size)
            } else {
                ms = benchmarkH2cCpuMsmBest(logSize: size)
            }

            // Update UI on main thread
            DispatchQueue.main.async {
                if isMetal {
                    lastMetalCPUResult = ms
                } else {
                    lastCPUOnlyResult = ms
                }
                // Unblock UI & stop timer
                stopTimer()
                isRunning = false
            }
        }
    }

    // MARK: - 100 Iterations Benchmark

    private func runHundredIterations() {
        // Block UI & start timer
        isRunning = true
        startTimer()

        DispatchQueue.global(qos: .userInitiated).async {
            let size = UInt32(logSize)

            var sumMetal: UInt64 = 0
            var sumCPU: UInt64 = 0

            for _ in 0..<100 {
                sumMetal += benchmarkH2cMetalAndCpuMsmBest(logSize: size)
                sumCPU += benchmarkH2cCpuMsmBest(logSize: size)
            }

            let avgMetal = Double(sumMetal) / 100.0
            let avgCpu   = Double(sumCPU) / 100.0

            DispatchQueue.main.async {
                avgMetalCPUResult = avgMetal
                avgCPUOnlyResult = avgCpu
                // Unblock UI & stop timer
                stopTimer()
                isRunning = false
            }
        }
    }

    // MARK: - Timer Management

    private func startTimer() {
        elapsedSeconds = 0
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            elapsedSeconds += 1
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

#Preview {
    ContentView()
}
