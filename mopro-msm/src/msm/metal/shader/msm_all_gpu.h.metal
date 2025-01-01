#pragma once

#include "curves/bn254.h.metal"
#include "fields/fp_bn254.h.metal"
#include "arithmetics/unsigned_int.h.metal"
#include "curves/ser_point.h.metal"

// Number of threads per threadgroup
// This is calculated based on max threadgroup shared memory size: ~ 32 KB / 96 bytes (size of a Point)
#define MAX_THREADS_PER_TG 340

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

    // 1) Choose your window size (C).
    //    For 25 points, C=4 or C=5 is typically reasonable.
    constexpr uint32_t C = 5;  // 5-bit window

    // 2) We'll assume 256-bit scalars, so numWindows = ceil(256 / C).
    //    If your scalar bit-size differs, adjust accordingly.
    constexpr uint32_t SCALAR_BITS = 256;
    const uint32_t numWindows = (SCALAR_BITS + C - 1) / C;

    // We'll also define a helper array for "buckets" in each window.
    // The number of possible bucket indices is 2^C - 1.
    Point window_buckets[(1 << C) - 1];
    // ^ If you do it in local thread memory,
    //   ensure (1 << C)-1 fits your memory constraints
    //   or you can store them in registers if that is simpler.

    // 3) Pippenger: loop over windows from LSB to MSB
    for (uint32_t w = 0; w < numWindows; w++) {

        // 3.1) Clear buckets for this window
        for (uint32_t b = 0; b < (1 << C) - 1; b++) {
            window_buckets[b] = Point::neutral_element();
        }
        // If you have multiple threads doing partial Pippenger,
        // you might need a barrier here, but if one thread does
        // all 25 pairs, it's just local.

        // 3.2) Accumulate each of our ~25 scalars/points into the correct bucket
        //     - Extract c bits from each scalar
        for (uint32_t i = thread_start; i < thread_end; i++) {
            // s_i = all_scalars[i], p_i = all_points[i]
            // Extract bits [w*C .. w*C + (C-1)] from s_i
            // => a small integer in [0 .. (1<<C)-1]

            // Some pseudo-code to extract bits:
            // (You can do shifting, masking, etc. with your big-int type.)
            u256 s_i = all_scalars[i];
            uint32_t shift_amount = w * C;

            // get the c-bit fragment:
            //   target_bucket = (s_i >> shift_amount) & ((1 << C) - 1)
            uint32_t target_bucket = s_i.extract_bits(shift_amount, C);

            //   e.g. if using a method that does: (s_i >> shift_amount).to_uint32() & ((1<<C)-1)

            // Now if target_bucket != 0, add p_i to bucket fragment-1
            if (target_bucket != 0) {
                // bucket index = fragment - 1
                uint32_t b_idx = target_bucket - 1;
                // accumulate p_i
                window_buckets[b_idx] = window_buckets[b_idx] + all_points[i];
            }
        }

        // 3.3) Now do a "sum reduction" of the buckets from high to low
        //     The typical trick:
        //     running_sum = 0
        //     for b in [ (1<<C)-1 .. 1 ]:
        //       running_sum += bucket[b-1]
        //       local_acc += running_sum
        Point running_sum = Point::neutral_element();
        for (int32_t b = (1 << C) - 2; b >= 0; b--) {
            running_sum = running_sum + window_buckets[b];
            local_acc = local_acc + running_sum;
        }

        // 3.4) Double local_acc by C bits for the next window
        //     (skip if w == numWindows-1)
        if (w < numWindows - 1) {;
            for (uint32_t dbl = 0; dbl < C; dbl++) {
                local_acc = local_acc.double_in_place();
            }
        }
    }

    // local_acc now holds the sum of all scalars * points

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