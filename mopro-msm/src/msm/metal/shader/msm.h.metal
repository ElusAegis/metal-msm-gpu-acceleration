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

    // skip if the scalar is uint scalar
    u256 one = u256::from_int((uint32_t)1);
    if (this_scalar == one) {
        return;
    }

    // for each window, record the corresponding bucket index and point idx
    for (uint32_t i = 0; i < num_windows; i++) {
        uint32_t window_idx = _window_starts[i];

        uint32_t scalar_fragment = (this_scalar >> window_idx).m_limbs[NUM_LIMBS - 1];
        uint32_t m_ij = scalar_fragment & buckets_len;

        // the case (b_idx, p_idx) = (0, 0) is not possible
        // since thread_id == 0 && i == 0 && m_ij == 1 is not possible
        if (m_ij != 0) {
            uint32_t bucket_idx = i * buckets_len + m_ij - 1;
            uint32_t point_idx = thread_id;
            buckets_indices[thread_id * num_windows + i] = uint2(bucket_idx, point_idx);
        }
    }
}

// TODO: sorting buckets_indices with bucket_idx as key

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
