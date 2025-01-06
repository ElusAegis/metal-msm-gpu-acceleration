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
    // Buffers/Params
    constant uint  & _instances_size      [[ buffer(0) ]],
    constant uint  & _num_windows         [[ buffer(1) ]],
    constant Point * p_buff               [[ buffer(2) ]],  // Array of all points
    device const uint2 * buckets_indices  [[ buffer(3) ]],  // (x,y) sorted by x
    device Point   * buckets_matrix       [[ buffer(4) ]],  // Output buckets
    constant uint  & _actual_threads      [[ buffer(5) ]],
    constant uint  & _total_buckets       [[ buffer(6) ]],

    // Shared memory for boundary accumulations
    threadgroup PairAccum *sharedLeftAccum  [[ threadgroup(0) ]],
    threadgroup PairAccum *sharedRightAccum [[ threadgroup(1) ]],

    // Threadgroup-related IDs
    uint t_id           [[ thread_index_in_threadgroup ]],
    uint group_id       [[ threadgroup_position_in_grid ]],
    uint threads_per_tg [[ threads_per_threadgroup ]]
)
{
    // -------------------------------------------------------------------------
    // Step 0: Basic checks
    // -------------------------------------------------------------------------
    uint totalBuckets = _total_buckets;
    if (totalBuckets == 0) {
        return;
    }
    uint globalThreadId = group_id * threads_per_tg + t_id;

    // total number of threadgroups
    // (Since you already compute _actual_threads outside, we can do the typical formula:)
    uint numThreadGroups = (_actual_threads + threads_per_tg - 1) / threads_per_tg;

    // The length of the pairs array (sorted by x).
    uint pairsLen = _instances_size * _num_windows;
    // If pairsLen == 0, nothing to do
    if (pairsLen == 0) {
        return;
    }

    // -------------------------------------------------------------------------
    // Step 1: Figure out which bucket-range belongs to this threadgroup
    // -------------------------------------------------------------------------
    // We'll compute how many buckets per group and get the range [bucketStart..bucketEnd).
    uint bucketsPerGroup = (totalBuckets + numThreadGroups - 1) / numThreadGroups;  // ceil
    uint bucketStart     = group_id * bucketsPerGroup;
    uint bucketEnd       = min(bucketStart + bucketsPerGroup, totalBuckets);

    if (bucketStart >= bucketEnd) {
        // This group is assigned an empty slice of buckets
        return;
    }

    // Next, we find in the `buckets_indices` array which segment corresponds
    // to [bucketStart, bucketEnd). We'll get [pairsGroupStart, pairsGroupEnd).
    uint pairsGroupStart = lower_bound_x(buckets_indices, pairsLen, bucketStart);
    uint pairsGroupEnd   = lower_bound_x(buckets_indices, pairsLen, bucketEnd);

    // If no pairs in this group range, bail
    if (pairsGroupStart >= pairsGroupEnd) {
        return;
    }

    // The total number of pairs that this entire workgroup must handle:
    uint pairsPerGroup = pairsGroupEnd - pairsGroupStart;

    // -------------------------------------------------------------------------
    // Step 2: Each thread's sub-fragment
    // -------------------------------------------------------------------------
    // We'll distribute these `pairsPerGroup` among all threads in the group
    uint pairsPerThread = (pairsPerGroup + threads_per_tg - 1) / threads_per_tg;  // ceil
    uint fragmentStart  = pairsGroupStart + t_id * pairsPerThread;
    uint fragmentEnd    = min(fragmentStart + pairsPerThread, pairsGroupEnd);
    uint neededThreads  = (pairsPerGroup + pairsPerThread - 1) / pairsPerThread;

    if (t_id >= neededThreads) {
        // This thread has no pairs to process
        sharedLeftAccum[t_id].x  = 0;
        sharedRightAccum[t_id].x = 0;
        sharedLeftAccum[t_id].val  = toSerBn254Point(Point::neutral_element());
        sharedRightAccum[t_id].val = toSerBn254Point(Point::neutral_element());
        return;
    }

    // -------------------------------------------------------------------------
    // Step 3: Local accumulation of sub-fragment
    // -------------------------------------------------------------------------
    // We'll track partial sums for each distinct x.
    // Keep track of the "first x" in this fragment and the "last x" in this fragment
    // so we can handle boundary merges.

    // We'll store partial sums for the "first x" and "last x" in this fragment
    // as we may need to merge with neighbors in the threadgroup.
    // For now, set them to neutral.
    PairAccum leftBoundary;
    leftBoundary.x  = buckets_indices[fragmentStart].x;
    leftBoundary.val = toSerBn254Point(Point::neutral_element());

    PairAccum rightBoundary;
    // Will be initialized later

    // Start reading from the first pair in this fragment
    uint2 initialPair = buckets_indices[fragmentStart];
    uint currentX = initialPair.x;
    Point localSum = p_buff[initialPair.y];

    // Iterate over the pairs in [fragmentStart..fragmentEnd)
    for (uint i = fragmentStart + 1; i < fragmentEnd; i++) {
        uint2 pairXY = buckets_indices[i];
        if (pairXY.x == currentX) {
            // same bucket => accumulate
            localSum += p_buff[pairXY.y];
        } else {
            // we've encountered a new bucket
            if (currentX == initialPair.x) {
                // This belongs to the left boundary (the first x in the fragment)
                // We'll store partial sum in leftBoundary, possibly to merge later
                leftBoundary.x  = currentX;
                leftBoundary.val = toSerBn254Point(localSum);
            } else {
                // This is a bucket fully contained in our fragment
                // so we can directly write it out to global memory,
                // because no other thread is processing this same x.
                buckets_matrix[currentX] = localSum;
            }
            // Move on to the new x
            currentX = pairXY.x;
            localSum = p_buff[pairXY.y];
        }
    }

    // After the loop ends, we have a partial sum for `currentX`.
    rightBoundary.x   = currentX;
    rightBoundary.val = toSerBn254Point(localSum);

    // Save boundaries to shared memory
    sharedLeftAccum[t_id]  = leftBoundary;
    sharedRightAccum[t_id] = rightBoundary;

    // -------------------------------------------------------------------------
    // Step 4: Synchronize Before Intra-Threadgroup Boundary Reduction
    // -------------------------------------------------------------------------
    threadgroup_barrier(mem_flags::mem_threadgroup);

//    // FIXME - remove this
//    uint placeholder = 1000000;
//    if (t_id < 32) {
//    buckets_matrix[group_id * 32 + t_id] = hide12ValuesIntoBN254Point(
//        group_id, globalThreadId, t_id, neededThreads,
//        pairsGroupStart, pairsGroupEnd, bucketStart, bucketEnd,
//        pairsPerThread, pairsPerGroup, leftBoundary.x, rightBoundary.x);
//        }
//    return;

    uint stride = 1;

    while (stride < threads_per_tg) {

        // Only threads whose ID satisfies the condition do merging
        if ((t_id % (2 * stride)) == 0 && (t_id + stride) < neededThreads) {
            uint leftIndex  = t_id;
            uint rightIndex = t_id + stride;

            // We have two pairs for each index:
            //   sharedLeftAccum[i],  sharedRightAccum[i]
            //   sharedLeftAccum[i+stride], sharedRightAccum[i+stride]
            PairAccum leftLeft   = sharedLeftAccum[leftIndex];
            PairAccum leftRight  = sharedRightAccum[leftIndex];
            PairAccum rightLeft  = sharedLeftAccum[rightIndex];
            PairAccum rightRight = sharedRightAccum[rightIndex];


            // ---------------------------------------------------------------------
            // 1) Depending on the relationships among these four boundaries,
            //    write partial sums to global if they are fully resolved,
            //    or merge into leftLeft / rightRight if needed.
            // ---------------------------------------------------------------------
            if (leftLeft.x != leftRight.x && rightRight.x != rightLeft.x) {
                // If leftRight.x == rightLeft.x is a distinct boundary,
                // we can immediately write to global
                if (leftRight.x == rightLeft.x) {
                    buckets_matrix[rightLeft.x] = fromSerBn254Point(leftRight.val) + fromSerBn254Point(rightLeft.val);
                } else {
                    buckets_matrix[leftRight.x] = fromSerBn254Point(leftRight.val);
                    buckets_matrix[rightLeft.x] = fromSerBn254Point(rightLeft.val);
                }
            } else if (leftLeft.x == rightLeft.x) {
                // We have a boundary collision (same bucket) that might also coincide
                // with leftLeft.x or rightRight.x. If so, we merge them.
                leftLeft.val = toSerBn254Point(fromSerBn254Point(leftLeft.val) + fromSerBn254Point(leftRight.val) + fromSerBn254Point(rightLeft.val));
            } else if (rightRight.x == leftRight.x) {
                 rightRight.val = toSerBn254Point(fromSerBn254Point(rightRight.val) + fromSerBn254Point(rightLeft.val) + fromSerBn254Point(leftRight.val));
            } else {
                // If the right-left boundary differs from left-right boundary
                // and neither merges with the other, we can write both out separately.
                if (leftLeft.x != leftRight.x) {
                    buckets_matrix[leftRight.x] = fromSerBn254Point(leftRight.val);
                } else {
                    leftLeft.val = toSerBn254Point(fromSerBn254Point(leftLeft.val) + fromSerBn254Point(leftRight.val));
                }
                if (rightRight.x != rightLeft.x) {
                    buckets_matrix[rightLeft.x] = fromSerBn254Point(rightLeft.val);
                } else {
                    rightRight.val = toSerBn254Point(fromSerBn254Point(rightRight.val) + fromSerBn254Point(rightLeft.val));
                }
            }

            // ---------------------------------------------------------------------
            // Finally, store updated results back into shared memory
            // so subsequent steps can see them. We only need to update left side.
            // ---------------------------------------------------------------------
            sharedLeftAccum[leftIndex]  = leftLeft;
            sharedRightAccum[leftIndex] = rightRight;



        }

        // Synchronize after each reduction step
        threadgroup_barrier(mem_flags::mem_threadgroup);

        stride *= 2;
    }

    // -----------------------------------------------------------------------------
    // Step 7: Final Write from Reduced Accumulators
    // (Same as pseudocode: only thread 0 merges the final leftover pair, if needed.)
    // -----------------------------------------------------------------------------
    if (t_id == 0) {
        PairAccum leftFinal  = sharedLeftAccum[t_id];
        PairAccum rightFinal = sharedRightAccum[t_id];

        if (leftFinal.x == rightFinal.x) {
            // Same bucket => merge them
            buckets_matrix[leftFinal.x] = fromSerBn254Point(leftFinal.val) + fromSerBn254Point(rightFinal.val);
        } else {
            buckets_matrix[leftFinal.x] = fromSerBn254Point(leftFinal.val);
            buckets_matrix[rightFinal.x] = fromSerBn254Point(rightFinal.val);
        }
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
