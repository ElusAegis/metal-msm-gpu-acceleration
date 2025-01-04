#pragma once

#include "curves/bn254.h.metal"
#include "fields/fp_bn254.h.metal"
#include "arithmetics/unsigned_int.h.metal"
#include "curves/ser_point.h.metal"

namespace {
    typedef UnsignedInteger<8> u256;
    typedef FpBN254 FieldElement;
    typedef ECPoint<FieldElement, 0, u256> Point;
}

constant constexpr uint32_t NUM_LIMBS = 8;  // u256

// instance-wise parallel
[[kernel]] void prepare_buckets_indices(
    constant const uint32_t& _window_size       [[ buffer(0) ]],
    constant const uint32_t* _window_starts     [[ buffer(1) ]],
    constant const uint32_t& _num_windows       [[ buffer(2) ]],
    constant const u256* k_buff                 [[ buffer(3) ]],
    device uint2* buckets_indices               [[ buffer(4) ]],
    const uint32_t thread_id                    [[ thread_position_in_grid ]],
    const uint32_t total_threads                [[ threads_per_grid ]]
)
{
    if (thread_id >= total_threads) {
        return;
    }

    uint32_t window_size = _window_size;    // c in arkworks code
    uint32_t num_windows = _num_windows;
    uint32_t buckets_len = (1 << window_size) - 1;
    u256 this_scalar = k_buff[thread_id];

    // for each window, record the corresponding bucket index and point idx
    for (uint32_t i = 0; i < num_windows; i++) {
        uint32_t window_idx = _window_starts[i];

        uint32_t scalar_fragment = (this_scalar >> window_idx).m_limbs[NUM_LIMBS - 1];
        uint32_t m_ij = scalar_fragment & buckets_len;


        // all the points that are not added are mapped to (0, 0) - uninitialized value
        // we need to handle this case in the next kernels
        // the case (b_idx, p_idx) = (0, 0) is ?not possible otherwise?
        // since thread_id == 0 && i == 0 && m_ij == 1 is not possible
        // TODO - is it really impossible? No, can happen if the lower window of a
        // scalar is 1. So we instead just add the m_ij = 0 as a special case
        if (m_ij != 0) {
            uint32_t bucket_idx = i * buckets_len + m_ij - 1;
            uint32_t point_idx = thread_id;
            buckets_indices[thread_id * num_windows + i] = uint2(bucket_idx, point_idx);
        } else {
            // TODO - we can avoid this case if we sort with when we get (0, 0) above
            buckets_indices[thread_id * num_windows + i] = uint2(-1, -1);
        }
    }
}

// Compare the x-values of two uint2 and swap if out of order.
inline void compare_and_swap_x(threadgroup uint2 &a, threadgroup uint2 &b, bool ascending) {
    // If ascending, swap if a.x > b.x.
    // If descending, swap if a.x < b.x.
    bool condition = ascending ? (a.x > b.x) : (a.x < b.x);
    if (condition) {
        uint2 tmp = a;
        a = b;
        b = tmp;
    }
}

[[kernel]] kernel void local_sort_buckets_indices(
    // The entire array of data to sort by .x ascending
    device uint2* data            [[ buffer(0) ]],

    // total_elems = total # of (uint2) elements in 'data'
    constant uint &total_elems    [[ buffer(1) ]],

    // block_size = 1024 in local sort pass, or bigger in merge passes
    constant uint &block_size     [[ buffer(2) ]],

    // We allocate local memory for up to 1024 elements
    // If block_size <= 1024, we can store them here for local sort.
    // For merges, we might store partial data or do repeated loading.
    threadgroup uint2* shared_data [[ threadgroup(0) ]],

    // Thread/group IDs
    uint tid             [[ thread_index_in_threadgroup ]],
    uint g_id            [[ threadgroup_position_in_grid ]],
    uint threads_per_tg  [[ threads_per_threadgroup ]]
)
{
    // The global "block" index in the array
    // Each block is 'block_size' elements
    // block_id = g_id (since we dispatch enough groups to cover total_elems / block_size)
    uint block_id = g_id;

    // The "global offset" in the array for this block
    uint block_start = block_id * block_size;
    if (block_start >= total_elems) {
        return; // no data
    }

    // The number of valid elements in this block (could be smaller at the end)
    uint block_count = min(block_size, total_elems - block_start);

    // Step 1: LOAD from global memory into threadgroup memory

    // If block_count < block_size, fill the remainder with a sentinel
    // if you want to handle partial blocks.
    for (uint i = tid; i < block_size; i += threads_per_tg) {
        if (i < block_count) {
            shared_data[i] = data[block_start + i];
        } else {
            // sentinel: set .x to 0xFFFFFFFF or so, if needed
            // or leave them as is if you can handle partial blocks.
            shared_data[i] = uint2(-1, -1);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Perform local sort or partial merges
    // For "stage=0, substage=0," do a full local bitonic sort up to 'block_count'.
    // For merges, we may do partial merges.
    // A simple approach is to do a standard bitonic build -> merge on block_size,
    // ignoring partials. We'll show a standard local bitonic:

    // We'll do the standard bitonic sequence in shared memory up to 'block_size'.
    // If block_count < block_size, some data might be sentinel. It's still okay.

    for (uint size = 2; size <= block_size; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {

            // Now loop i in steps of threads_per_tg, starting from tid
            for (uint i = tid; i < block_size; i += threads_per_tg) {
                uint partner = i ^ stride;
                if (partner > i && partner < block_size) {
                    bool ascending = ((i & size) == 0);
                    compare_and_swap_x(shared_data[i], shared_data[partner], ascending);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Step 3: STORE back to global
    // Each thread writes if within block_count
    for (uint i = tid; i < block_size; i += threads_per_tg) {
        if (i < block_count) {
            data[block_start + i] = shared_data[i];
        }
    }
}

/// Kernel to merge two sorted sub-blocks into a single sorted block using tile-based merging
[[kernel]] void merge_sort_buckets_indices(
    // The input buffer holding two sorted sub-blocks
    device const uint2* buffer_in          [[ buffer(0) ]],
    // The output buffer to write the merged results
    device uint2*       buffer_out         [[ buffer(1) ]],

    // Total number of (uint2) elements in the dataset
    constant uint &total_elems             [[ buffer(2) ]],
    // The size of each sorted sub-block we are merging
    constant uint &curr_block_size         [[ buffer(3) ]],

    // Shared memory used for partial merging
    // We'll store up to tile_size items from A and B, and also a "temp" array
    // so the total needed = tile_size * 2 `uint2`s = tile_size * 2 * 8 bytes
    threadgroup uint2* shared_data         [[ threadgroup(0) ]],

    // Standard thread / group identifiers
    uint tid             [[ thread_index_in_threadgroup ]],
    uint group_id        [[ threadgroup_position_in_grid ]],
    uint threads_per_tg  [[ threads_per_threadgroup ]]
)
{
    // Each threadgroup merges one pair of sub-blocks:
    //   Sub-block A: [start, start + curr_block_size)
    //   Sub-block B: [start + curr_block_size, start + 2*curr_block_size)
    uint pair_id = group_id;
    uint start   = pair_id * (curr_block_size * 2);

    // If we exceed the total elements, nothing to merge
    if (start >= total_elems) {
        return;
    }

    // Actual number of elements in each sub-block, in case we are near the end
    uint A_count = min(curr_block_size, total_elems - start);
    uint B_count = 0;
    if (A_count < curr_block_size) {
        // means we are near the end
        B_count = 0;
    } else {
        // sub-block B is directly after sub-block A
        B_count = min(curr_block_size, total_elems - (start + A_count));
    }

    // We'll define a tile_size. Let's pick 256 for demonstration, but it must
    // match your shared memory usage constraints. For testing, you used 4.
    const uint tile_size = 4; // <= 2 * tile_size * 8 bytes must fit in shared memory

    // We'll interpret:
    //   the first tile_size positions in shared_data as "tileAB" for loaded input
    //   the next tile_size positions in shared_data as "temp" for merged output
    // so we need `2 * tile_size` in shared memory.

    uint2 threadgroup* tileAB  = shared_data;              // tileAB[0..(tile_size-1)]
    uint2 threadgroup* tileTmp = shared_data + tile_size;  // tileTmp[0..(tile_size-1)]

    // We'll track positions in sub-block A (A_pos), sub-block B (B_pos),
    // and where we place merged data in the global output (out_pos).
    uint A_pos   = 0;
    uint B_pos   = 0;
    uint out_pos = 0;

    // While we still have data in A or B
    while (A_pos < A_count || B_pos < B_count) {
        // For each "tile", load up to tile_size/2 from A, tile_size/2 from B
        // ensuring we don't exceed sub-block sizes
        uint loadA = min(tile_size / 2, A_count - A_pos);
        uint loadB = min(tile_size / 2, B_count - B_pos);

        // Load sub-block A slice into tileAB[0..(loadA-1)]
        for (uint i = tid; i < loadA; i += threads_per_tg) {
            tileAB[i] = buffer_in[start + A_pos + i];
        }
        // Fill remainder of left half with sentinel
        for (uint i = loadA + tid; i < tile_size / 2; i += threads_per_tg) {
            tileAB[i] = uint2(0xFFFFFFFF, 0xFFFFFFFF);
        }

        // Load sub-block B slice into tileAB[tile_size/2..(tile_size/2 + loadB -1)]
        for (uint i = tid; i < loadB; i += threads_per_tg) {
            tileAB[tile_size / 2 + i] = buffer_in[start + A_count + B_pos + i];
        }
        // Fill remainder of right half with sentinel
        for (uint i = loadB + tid; i < tile_size / 2; i += threads_per_tg) {
            tileAB[tile_size / 2 + i] = uint2(0xFFFFFFFF, 0xFFFFFFFF);
        }

        // Barrier to ensure tileAB is fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Now do a direct 2-way merge in local memory:
        // tileAB[0..(tile_size/2 -1)] sorted ascending, tileAB[tile_size/2..(tile_size -1)] sorted ascending
        // We'll produce the merged result in tileTmp[0..(loadA+loadB -1)]

        uint left_idx  = 0;
        uint right_idx = tile_size / 2;
        uint merge_end = loadA + loadB;  // how many valid items we loaded
//        for (uint i = tid; i < merge_end; i += threads_per_tg) {
//            // We'll do a normal "two-pointer" approach:
//            // if left is not exhausted AND (right is exhausted OR left <= right):
//            //   tileTmp[i] = tileAB[left_idx++]
//            // else tileTmp[i] = tileAB[right_idx++]
//        }

        // We'll do that in a loop so each thread can handle multiple merges
        uint i = tid;
        // Do that a single thread with id = 0 does the merge sort fo the two tiles
        if (tid == 0) {
            while (i < merge_end) {
                // Because each iteration merges exactly one element, we want a single pass approach
                // Let's just do an atomic approach in each thread. We'll do a single pass with a loop:

                // We'll define a local pass variable so we distribute merges across threads
                // Actually, simpler is to do a single for i in [0..merge_end) with i += threads_per_tg.

                // Pseudocode for each iteration:
                //   if (left_idx < loadA && (right_idx >= tile_size || tileAB[left_idx].x <= tileAB[right_idx].x)) {
                //       tileTmp[i] = tileAB[left_idx];
                //       left_idx++;
                //   } else {
                //       tileTmp[i] = tileAB[right_idx];
                //       right_idx++;
                //   }

                if (left_idx < loadA && (right_idx >= tile_size || tileAB[left_idx].x <= tileAB[right_idx].x)) {
                    tileTmp[i] = tileAB[left_idx];
                    left_idx++;
                } else {
                    tileTmp[i] = tileAB[right_idx];
                    right_idx++;
                }
                i += 1;
            }
        }
//        while (i < merge_end) {
//            // Because each iteration merges exactly one element, we want a single pass approach
//            // Let's just do an atomic approach in each thread. We'll do a single pass with a loop:
//
//            // We'll define a local pass variable so we distribute merges across threads
//            // Actually, simpler is to do a single for i in [0..merge_end) with i += threads_per_tg.
//
//            // Pseudocode for each iteration:
//            //   if (left_idx < loadA && (right_idx >= tile_size || tileAB[left_idx].x <= tileAB[right_idx].x)) {
//            //       tileTmp[i] = tileAB[left_idx];
//            //       left_idx++;
//            //   } else {
//            //       tileTmp[i] = tileAB[right_idx];
//            //       right_idx++;
//            //   }
//
//            if (left_idx < loadA && (right_idx >= tile_size || tileAB[left_idx].x <= tileAB[right_idx].x)) {
//                tileTmp[i] = tileAB[left_idx];
//                left_idx++;
//            } else {
//                tileTmp[i] = tileAB[right_idx];
//                right_idx++;
//            }
//            i += threads_per_tg;
//        }

        // Barrier so that tileTmp is consistent
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write tileTmp[0..to_write-1] to global buffer
        uint to_write = loadA + loadB; // total merged in this tile
//        uint remain   = (A_count + B_count) - out_pos;
//        to_write = (to_write < remain) ? to_write : remain;

        for (uint i2 = tid; i2 < to_write; i2 += threads_per_tg) {
            buffer_out[start + out_pos + i2] = tileTmp[i2];
        }

        // Update positions
        A_pos   += loadA;
        B_pos   += loadB;
        out_pos += to_write;

        // Next tile...
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

uint lower_bound_x(device const uint2 *arr, uint n, uint key) {
    uint left = 0;
    uint right = n;
    while (left < right) {
        uint mid = (left + right) >> 1;
        if (arr[mid].x < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

[[kernel]] void bucket_wise_accumulation(
    constant uint & _instances_size     [[ buffer(0) ]],
    constant uint & _num_windows        [[ buffer(1) ]],
    constant Point * p_buff             [[ buffer(2) ]],
    device   uint2 * buckets_indices    [[ buffer(3) ]],
    device   Point * buckets_matrix     [[ buffer(4) ]],
    constant uint & _actual_threads     [[ buffer(5) ]],

    // Threadgroup storage for serialized partial sums
    threadgroup SerBn254Point *shared_accumulators [[ threadgroup(0) ]],

    // Threadgroup-related IDs
    uint t_id             [[ thread_index_in_threadgroup ]],
    uint group_id         [[ threadgroup_position_in_grid ]],
    uint threads_per_tg   [[ threads_per_threadgroup ]]
)
{
    // 1) total number of buckets
    uint total_buckets = _instances_size * _num_windows;
    if (total_buckets == 0) {
        return;
    }

    // 2) Our global thread ID
    uint global_id = group_id * threads_per_tg + t_id;
    if (global_id >= _actual_threads) {
        // This thread does nothing
        return;
    }

    // 3) Partition the [0..total_buckets) space among _actual_threads
    uint chunk_size   = (total_buckets + _actual_threads - 1) / _actual_threads;
    uint start_bucket = global_id * chunk_size;
    uint end_bucket   = min(start_bucket + chunk_size, total_buckets);
    if (start_bucket >= end_bucket) {
        // No buckets to handle
        return;
    }

    // 4) If you need an explicit length for `buckets_indices`,
    //    define or pass it. Often it’s just the length of the array.
    uint indices_len = _instances_size * _num_windows; // or your actual list length

    // 5) For each bucket in [start_bucket..end_bucket)
    for (uint b = start_bucket; b < end_bucket; b++) {
        // 5a) Find the subrange of `buckets_indices` for .x == b
        uint i1 = lower_bound_x(buckets_indices, indices_len, b);
        uint i2 = lower_bound_x(buckets_indices, indices_len, b + 1);
        if (i1 >= i2) {
            // No occurrences of this bucket ID
            // But do barrier to keep group in sync for next bucket
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // 5b) Number of points that map to this bucket
        uint count_for_bucket = i2 - i1;

        // 5c) We want each thread to handle up to 100 additions
        const uint chunk_per_thread = 100;
        uint needed_threads = (count_for_bucket + chunk_per_thread - 1) / chunk_per_thread;

        // 5d) This thread only does work if t_id < needed_threads
        Point local_sum = Point::neutral_element();
        if (t_id < needed_threads) {
            // Determine this thread’s subrange in [i1..i2)
            uint start_idx = i1 + t_id * chunk_per_thread;
            uint end_idx   = min(start_idx + chunk_per_thread, i2);

            // Accumulate partial sum
            for (uint idx = start_idx; idx < end_idx; idx++) {
                uint yIndex = buckets_indices[idx].y;
                local_sum = local_sum + p_buff[yIndex];
            }
        }

        // 5e) Serialize the local sum and store in threadgroup memory
        shared_accumulators[t_id] = toSerBn254Point(local_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5f) Parallel reduction among the first `needed_threads` threads
        uint reduce_count = (needed_threads < threads_per_tg)
                            ? needed_threads
                            : threads_per_tg;

        for (uint offset = reduce_count >> 1; offset > 0; offset >>= 1) {
            if (t_id < offset) {
                Point p1 = fromSerBn254Point(shared_accumulators[t_id]);
                Point p2 = fromSerBn254Point(shared_accumulators[t_id + offset]);
                shared_accumulators[t_id] = toSerBn254Point(p1 + p2);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // 5g) Thread 0 updates buckets_matrix[b]
        if (t_id == 0) {
            Point final_val = fromSerBn254Point(shared_accumulators[0]);
            buckets_matrix[b] = final_val;
        }

        // 5h) Barrier before next bucket iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// window-wise reduction
[[kernel]] void sum_reduction(
    constant uint32_t &window_size        [[ buffer(0) ]],
    constant u256*    k_buff              [[ buffer(1) ]],
    constant Point*   p_buff              [[ buffer(2) ]],
    device   Point*   buckets_matrix      [[ buffer(3) ]],
    device   Point*   res                 [[ buffer(4) ]],

    // Threadgroup storage for partial sums
    threadgroup SerBn254Point *shared_accumulators [[ threadgroup(0) ]],

    // Thread/group IDs
    uint t_id        [[ thread_index_in_threadgroup ]],
    uint group_id    [[ threadgroup_position_in_grid ]],
    uint threadcount [[ threads_per_threadgroup ]]
)
{
    uint32_t c = window_size;                   // Window exponent
    uint32_t buckets_len = (1 << c) - 1;        // # of buckets for each window
    if (group_id >= window_size) return;  // Just a safety check, might not be needed

    // Each thread group corresponds to one window "j = group_id"
    uint32_t window_base_idx = group_id * buckets_len;

//    // Optionally: check if the scalar for window j is '1' => add base point
//    // TODO - can this be removed?
//    if (t_id == 0) {
//        // If your 'k_buff' usage is relevant:
//        u256 one_val = u256::from_int((uint32_t) 1);
//        u256 k = k_buff[group_id];
//
//        if (k == one_val) {
//            // Just initialize res[group_id] by adding p_buff[group_id] once
//            res[group_id] = Point::neutral_element() + p_buff[group_id];
//        } else {
//            res[group_id] = Point::neutral_element();
//        }
//    }

    // Divide the buckets among threads within the group
    uint32_t chunk_size = (buckets_len + threadcount - 1) / threadcount;
    uint32_t start_idx  = t_id * chunk_size;
    uint32_t end_idx    = (start_idx + chunk_size < buckets_len)
                          ? (start_idx + chunk_size)
                          : buckets_len;

    // Compute partial sums in this thread
    Point local_sum = Point::neutral_element();
    for (uint32_t i = start_idx; i < end_idx; i++) {
        local_sum = local_sum + buckets_matrix[window_base_idx + i];
    }

    // Write partial sum into threadgroup memory
    shared_accumulators[t_id] = toSerBn254Point(local_sum);

    // Synchronize so all threads have finished writing partial sums
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction of partial sums in shared memory
    for (uint32_t offset = threadcount / 2; offset > 0; offset >>= 1) {
        if (t_id < offset) {
            Point local_accumulator_1 = fromSerBn254Point(shared_accumulators[t_id]);
            Point local_accumulator_2 = fromSerBn254Point(shared_accumulators[t_id + offset]);
            Point combined = local_accumulator_1 + local_accumulator_2;
            shared_accumulators[t_id] = toSerBn254Point(combined);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes final sum for this window
    if (t_id == 0) {
        // Add the reduced bucket sum to the window’s existing value in `res[group_id]`.
        // If you do NOT want to accumulate, just assign. Otherwise, do an addition:
        Point final_acc = fromSerBn254Point(shared_accumulators[0]);
        Point this_res = res[group_id];
        res[group_id] = this_res + final_acc;
    }
}


[[kernel]] void final_accumulation(
    constant const uint32_t& _window_size       [[ buffer(0) ]],
    constant const uint32_t* _window_starts     [[ buffer(1) ]],
    constant const uint32_t& _num_windows       [[ buffer(2) ]],
    device Point* res                           [[ buffer(3) ]],
    device Point& msm_result                    [[ buffer(4) ]]
)
{
    uint32_t window_size = _window_size;    // c in arkworks code
    uint32_t num_windows = _num_windows;
    Point lowest_window_sum = res[0];
    uint32_t last_res_idx = num_windows - 1;

    Point total_sum = Point::neutral_element();
    for (uint32_t i = 1; i < num_windows; i++) {
        Point tmp = total_sum;
        total_sum = tmp + res[last_res_idx - i + 1];

        for (uint32_t j = 0; j < window_size; j++) {
            total_sum = total_sum.double_in_place();
        }
    }
    msm_result = total_sum + lowest_window_sum;
}
