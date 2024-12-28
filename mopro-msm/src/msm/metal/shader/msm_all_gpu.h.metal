#pragma once

#include "curves/bn254.h.metal"
#include "fields/fp_bn254.h.metal"
#include "arithmetics/unsigned_int.h.metal"
#include "arithmetics/pair.h.metal"

// Number of threads per threadgroup
// This is calculated based on max threadgroup shared memory size: ~ 32 KB / 96 bytes (size of a Point)
#define MAX_THREADS_PER_TG 256

// Example definitions; adapt as needed
namespace {
    typedef UnsignedInteger<8> u256;               // 256-bit scalar
    typedef FpBN254 FieldElement;                  // BN254 field
    // We assume Point here is 96 bytes total: 3 × 32 bytes (x, y, z).
    // The third template param <u256> ensures it can handle multiplication by 256-bit scalars.
    typedef ECPoint<FieldElement, 0, u256> Point;
}


// Some Apple M-series devices have ~32 KB threadgroup memory.
// The user sets threads_per_threadgroup so that
//  threads_per_threadgroup * 128 (bytes) <= 32 KB
// for the final partial accumulators in shared memory.

[[kernel]]
void msm_all_gpu(
    // 1) Input buffers
    device const Point*   all_points    [[ buffer(0) ]],
    device const u256*    all_scalars   [[ buffer(1) ]],

    // 2) Size / indexing info
    constant const uint32_t& total_size [[ buffer(2) ]],
    // Number of scalar-point pairs each thread will process locally
    constant const uint32_t& batch_size [[ buffer(3) ]],

    //   How many threadgroups * threads each kernel launch has, etc.

    // 3) Output partial sums (one per threadgroup)
    device Point* partial_results       [[ buffer(4) ]],

    // 4) Threadgroup dispatch info
    uint   t_id                         [[ thread_index_in_threadgroup ]],
    uint   tg_id                        [[ threadgroup_position_in_grid ]],
    uint   threads_per_tg               [[ threads_per_threadgroup ]]
)
{
    ////////////////////////////////////////////////////////////////////////////
    // 0. Each threadgroup processes a distinct chunk of all_points/all_scalars
    //    The chunk size = threads_per_tg * batch_size.
    ////////////////////////////////////////////////////////////////////////////

    uint32_t chunk_size = threads_per_tg * batch_size;
    uint32_t chunk_offset = tg_id * chunk_size;

    // If chunk_offset >= total_size, nothing to do
    if (chunk_offset >= total_size) {
        return;
    }

    // The actual number of pairs in this chunk
    uint32_t chunk_end = min(chunk_offset + chunk_size, total_size);
    uint32_t chunk_len = chunk_end - chunk_offset; // might be partial for the last group

    ////////////////////////////////////////////////////////////////////////////
    // 1. Local Accumulator: Each thread accumulates batch_size pairs (or fewer).
    ////////////////////////////////////////////////////////////////////////////

    // Start with a neutral point
    Point local_acc = Point::neutral_element();

    // Compute the start for this thread inside the chunk
    uint32_t thread_start = chunk_offset + (t_id * batch_size);
    uint32_t thread_end   = min(thread_start + batch_size, chunk_offset + chunk_len);

    // For code clarity, just do a simple loop
    for (uint32_t i = thread_start; i < thread_end; i++) {
        // We have direct access to:  Point operator*(scalar)
        // Let p = all_points[i], s = all_scalars[i]
        Point local_point = all_points[i];
        u256 local_scalar = all_scalars[i];

        local_acc = local_acc + (local_point * local_scalar);
    }

    ////////////////////////////////////////////////////////////////////////////
    // 2. Threadgroup Shared Memory + Synchronization
    ////////////////////////////////////////////////////////////////////////////

    // We will store each thread's local_acc in shared memory so we can reduce
    // Because the threadgroups can not store structs with complex initialization
    // We create a custom `SerBn254Point` struct to store values
    threadgroup SerBn254Point shared_accumulators[MAX_THREADS_PER_TG];

    // Write local accumulator to shared memory
    shared_accumulators[t_id] = toSerBn254Point(local_acc);

    // Ensure all threads have finished writing
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ////////////////////////////////////////////////////////////////////////////
    // 3. Intra-Threadgroup Reduction
    //    Pairwise reduce accumulators until only one partial sum remains.
    ////////////////////////////////////////////////////////////////////////////

    // For a standard tree-based reduction over threads:
    // e.g., step = 1, 2, 4, 8, ...
    // at each step, thread t merges with t + step if in range, then we barrier.
    // We'll do log2(threads_per_tg) steps.
    // This code is simplistic and uses the entire array each pass.

    uint32_t reduction_count = threads_per_tg;
    uint32_t step = 1;

    while (reduction_count > 1) {
        // cur_half = the number of pairs we can merge at this step
        uint32_t cur_half = (reduction_count + 1) >> 1; // round up if odd

        // Only threads < cur_half do merging
        if (t_id < cur_half && (t_id + cur_half) < reduction_count) {
            // Add the partial from (t_id + cur_half) into current thread
            Point local_accumulator_1 = fromSerBn254Point(shared_accumulators[t_id]);
            Point local_accumulator_2 = fromSerBn254Point(shared_accumulators[t_id + cur_half]);
            shared_accumulators[t_id] = toSerBn254Point(local_accumulator_1 + local_accumulator_2);
        }

        // Synchronize so that all merges at this step are visible
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Next iteration merges half as many
        reduction_count = cur_half;
        step <<= 1;
    }

    // After this loop, thread 0 has the combined partial for the entire chunk
    if (t_id == 0) {
        partial_results[tg_id] = fromSerBn254Point(shared_accumulators[0]);
    }
}