use rand::RngCore;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};
use std::ops::Add;

pub trait ToLimbs<const N: usize> {
    /// Writes this object’s limbs directly into `out`.
    /// No allocations, minimal overhead.
    fn write_u32_limbs(&self, out: &mut [u32]);

    /// Default method: returns a `[u32; N]` by internally calling `write_u32_limbs`.
    fn to_u32_limbs(&self) -> Vec<u32> {
        let mut arr = [0u32; N];
        self.write_u32_limbs(&mut arr);
        arr.to_vec()
    }
}

impl<P, const N: usize> ToLimbs<N> for &[P]
where
    P: ToLimbs<N> + Sync,
{
    fn write_u32_limbs(&self, out: &mut [u32]) {
        assert_eq!(out.len(), self.len() * N, "Output buffer size is incorrect");

        // Use Rayon to parallelize over larger chunks
        self.par_iter()
            .zip(out.par_chunks_mut(N))
            .for_each(|(item, chunk)| {
                item.write_u32_limbs(chunk);
            });
    }

    /// Default method: returns a `[u32; N]` by internally calling `write_u32_limbs`.
    fn to_u32_limbs(&self) -> Vec<u32> {
        let mut vec = vec![0u32; self.len() * N];
        self.write_u32_limbs(vec.as_mut_slice());
        vec
    }
}

pub trait FromLimbs {
    const U32_SIZE: usize;

    fn from_u32_limbs(limbs: &[u32]) -> Self;
}

pub trait ScalarGPU<const N: usize>: ToLimbs<N> {
    const N: usize = N;
    const MODULUS_BIT_SIZE: usize;

    fn random(rng: &mut impl RngCore) -> Self;

    fn into<B: ScalarGPU<N> + FromLimbs>(&self) -> B {
        assert_eq!(
            Self::MODULUS_BIT_SIZE,
            B::MODULUS_BIT_SIZE,
            "Incompatible scalar sizes"
        );
        B::from_u32_limbs(&self.to_u32_limbs())
    }
}

pub trait PointGPU<const N: usize>:
    FromLimbs + ToLimbs<N> + Add<Self, Output = Self> + Sized + Clone
{
    fn random(rng: &mut impl RngCore) -> Self;

    fn into<B: PointGPU<N>>(&self) -> B {
        assert_eq!(Self::U32_SIZE, B::U32_SIZE, "Incompatible point sizes");
        B::from_u32_limbs(&self.to_u32_limbs())
    }
}

#[cfg(feature = "ark")]
pub mod ark {
    use super::{FromLimbs, PointGPU, ScalarGPU, ToLimbs};
    use ark_ff::{BigInteger256, PrimeField};

    pub use ark_bn254::{Fq as ArkFq, Fr as ArkFr, G1Affine as ArkGAffine, G1Projective as ArkG};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_std::UniformRand;
    use rand::RngCore;

    // convert from little endian to big endian
    impl ToLimbs<8> for BigInteger256 {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            // self.0 is [u64; 4], with self.0[0] being the least significant limb
            let [a0, a1, a2, a3] = self.0;

            // We want big-endian output:
            // out[0..2] from the highest 64 bits (a3)
            // out[2..4] from a2
            // out[4..6] from a1
            // out[6..8] from a0 (lowest 64 bits)
            out[0] = (a3 >> 32) as u32;
            out[1] = (a3 & 0xFFFF_FFFF) as u32;
            out[2] = (a2 >> 32) as u32;
            out[3] = (a2 & 0xFFFF_FFFF) as u32;
            out[4] = (a1 >> 32) as u32;
            out[5] = (a1 & 0xFFFF_FFFF) as u32;
            out[6] = (a0 >> 32) as u32;
            out[7] = (a0 & 0xFFFF_FFFF) as u32;
        }
    }

    // convert from little endian to big endian
    impl ToLimbs<8> for ArkFq {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            // Just delegate to the underlying BigInteger256
            self.0.write_u32_limbs(out);
        }
    }

    impl ToLimbs<8> for ArkFr {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            // Convert self => BigInteger256 => write
            self.into_bigint().write_u32_limbs(out);
        }
    }

    impl ToLimbs<24> for ArkG {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            // x, y, z => each is 8 limbs
            self.x.write_u32_limbs(&mut out[0..8]);
            self.y.write_u32_limbs(&mut out[8..16]);
            self.z.write_u32_limbs(&mut out[16..24]);
        }
    }

    impl ToLimbs<24> for ArkGAffine {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            // Convert affine -> projective => write
            self.into_group().write_u32_limbs(out);
        }
    }

    impl FromLimbs for BigInteger256 {
        const U32_SIZE: usize = 8;

        // convert from big endian to little endian for metal
        fn from_u32_limbs(limbs: &[u32]) -> Self {
            let mut big_int = [0u64; 4];
            for (i, limb) in limbs.chunks(2).rev().enumerate() {
                let high = u64::from(limb[0]);
                let low = u64::from(limb[1]);
                big_int[i] = (high << 32) | low;
            }
            BigInteger256::new(big_int)
        }
    }

    impl FromLimbs for ArkFq {
        const U32_SIZE: usize = 8;

        // convert from big endian to little endian for metal
        fn from_u32_limbs(limbs: &[u32]) -> Self {
            ArkFq::new_unchecked(BigInteger256::from_u32_limbs(limbs))
        }
    }

    impl FromLimbs for ArkFr {
        const U32_SIZE: usize = 8;

        fn from_u32_limbs(limbs: &[u32]) -> Self {
            Self::from(BigInteger256::from_u32_limbs(limbs))
        }
    }

    impl ScalarGPU<8> for ArkFr {
        const MODULUS_BIT_SIZE: usize = <ArkFr as PrimeField>::MODULUS_BIT_SIZE as usize;

        fn random(rng: &mut impl RngCore) -> Self {
            <Self as UniformRand>::rand(rng)
        }
    }

    impl FromLimbs for ArkG {
        const U32_SIZE: usize = 24;

        fn from_u32_limbs(limbs: &[u32]) -> Self {
            Self::new_unchecked(
                ArkFq::from_u32_limbs(&limbs[0..8]),
                ArkFq::from_u32_limbs(&limbs[8..16]),
                ArkFq::from_u32_limbs(&limbs[16..24]),
            )
        }
    }

    impl PointGPU<24> for ArkG {
        fn random(rng: &mut impl RngCore) -> Self {
            <Self as UniformRand>::rand(rng).into_affine().into_group()
        }
    }
}

#[cfg(feature = "h2c")]
pub mod h2c {
    use super::{FromLimbs, PointGPU, ScalarGPU, ToLimbs};
    use halo2curves::CurveExt;

    pub use halo2curves::bn256::{Fq as H2Fq, Fr as H2Fr, G1Affine as H2GAffine, G1 as H2G};
    use halo2curves::ff::Field;
    use halo2curves::group::prime::PrimeCurveAffine;
    use halo2curves::group::{Curve, Group};
    use halo2curves::serde::SerdeObject;
    use rand::RngCore;

    use std::io::Write;

    /// A custom writer that writes directly into a fixed-size byte buffer.
    struct SliceWriter<'a> {
        buffer: &'a mut [u8; 32],
        pos: usize,
    }

    impl<'a> SliceWriter<'a> {
        /// Creates a new `SliceWriter` instance.
        fn new(buffer: &'a mut [u8; 32]) -> Self {
            SliceWriter { buffer, pos: 0 }
        }
    }

    impl Write for SliceWriter<'_> {
        fn write(&mut self, buf: &[u8]) -> std::result::Result<usize, std::io::Error> {
            let available = self.buffer.len() - self.pos;
            let to_write = buf.len().min(available);
            self.buffer[self.pos..self.pos + to_write].copy_from_slice(&buf[..to_write]);
            self.pos += to_write;
            Ok(to_write)
        }

        fn flush(&mut self) -> std::result::Result<(), std::io::Error> {
            Ok(())
        }
    }

    #[inline(always)]
    fn bytes_to_u32_reverse(input: &[u8], out: &mut [u32]) {
        out[0] = (input[28] as u32)
            | ((input[29] as u32) << 8)
            | ((input[30] as u32) << 16)
            | ((input[31] as u32) << 24);

        out[1] = (input[24] as u32)
            | ((input[25] as u32) << 8)
            | ((input[26] as u32) << 16)
            | ((input[27] as u32) << 24);

        out[2] = (input[20] as u32)
            | ((input[21] as u32) << 8)
            | ((input[22] as u32) << 16)
            | ((input[23] as u32) << 24);

        out[3] = (input[16] as u32)
            | ((input[17] as u32) << 8)
            | ((input[18] as u32) << 16)
            | ((input[19] as u32) << 24);

        out[4] = (input[12] as u32)
            | ((input[13] as u32) << 8)
            | ((input[14] as u32) << 16)
            | ((input[15] as u32) << 24);

        out[5] = (input[8] as u32)
            | ((input[9] as u32) << 8)
            | ((input[10] as u32) << 16)
            | ((input[11] as u32) << 24);

        out[6] = (input[4] as u32)
            | ((input[5] as u32) << 8)
            | ((input[6] as u32) << 16)
            | ((input[7] as u32) << 24);

        out[7] = (input[0] as u32)
            | ((input[1] as u32) << 8)
            | ((input[2] as u32) << 16)
            | ((input[3] as u32) << 24);
    }

    impl ToLimbs<8> for H2Fr {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            let input = self.to_bytes(); // 32 bytes
                                         // Fill out[] in reverse order of 4-byte chunks
            bytes_to_u32_reverse(&input, out);
        }
    }

    impl ToLimbs<8> for H2Fq {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            // Ensure the output buffer has enough space
            debug_assert!(
                out.len() >= 8,
                "Output buffer must have at least 8 elements"
            );

            // Create a fixed-size byte array
            let mut buffer = [0u8; 32];

            // Create a SliceWriter to write directly into the buffer
            {
                let mut writer = SliceWriter::new(&mut buffer);
                self.write_raw(&mut writer)
                    .expect("Failed to write raw bytes");
            }

            // Convert the byte buffer to limbs using the optimized utility function
            bytes_to_u32_reverse(&buffer, out);
        }
    }

    impl ToLimbs<24> for H2G {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            self.x.write_u32_limbs(&mut out[0..8]);
            self.y.write_u32_limbs(&mut out[8..16]);
            self.z.write_u32_limbs(&mut out[16..24]);
        }
    }

    impl ToLimbs<24> for H2GAffine {
        fn write_u32_limbs(&self, out: &mut [u32]) {
            self.x.write_u32_limbs(&mut out[0..8]);
            self.y.write_u32_limbs(&mut out[8..16]);
            H2Fq::one().write_u32_limbs(&mut out[16..24]);
        }
    }

    impl FromLimbs for H2Fr {
        const U32_SIZE: usize = 8;

        fn from_u32_limbs(limbs: &[u32]) -> Self {
            let mut big_int = [0u64; 4];
            for (i, limb) in limbs.chunks(2).rev().enumerate() {
                let high = u64::from(limb[0]);
                let low = u64::from(limb[1]);
                big_int[i] = (high << 32) | low;
            }
            Self::from_raw(big_int)
        }
    }

    impl ScalarGPU<8> for H2Fr {
        const MODULUS_BIT_SIZE: usize = 254;

        fn random(rng: &mut impl RngCore) -> Self {
            <Self as Field>::random(rng)
        }
    }

    impl FromLimbs for H2Fq {
        const U32_SIZE: usize = 8;

        fn from_u32_limbs(limbs: &[u32]) -> Self {
            let mut output = [0u8; 32];
            for (i, &value) in limbs.iter().rev().enumerate() {
                let bytes = value.to_le_bytes(); // Convert u32 to little-endian bytes
                output[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }
            Self::from_raw_bytes(&output).unwrap()
        }
    }

    impl FromLimbs for H2G {
        const U32_SIZE: usize = 24;

        fn from_u32_limbs(limbs: &[u32]) -> Self {
            H2G::new_jacobian(
                H2Fq::from_u32_limbs(&limbs[0..8]),
                H2Fq::from_u32_limbs(&limbs[8..16]),
                H2Fq::from_u32_limbs(&limbs[16..24]),
            )
            .expect("Failed to create `new_jacobian` point from limbs")
        }
    }

    impl FromLimbs for H2GAffine {
        const U32_SIZE: usize = 24;

        fn from_u32_limbs(limbs: &[u32]) -> Self {
            H2G::from_u32_limbs(limbs).to_affine()
        }
    }

    impl PointGPU<24> for H2G {
        fn random(rng: &mut impl RngCore) -> Self {
            <Self as Group>::random(rng).to_affine().to_curve()
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(unused_imports)]

    use super::{FromLimbs, ToLimbs};
    #[cfg(feature = "ark")]
    use crate::metal::abstraction::limbs_conversion::ark::{ArkFq, ArkFr, ArkG};
    #[cfg(feature = "h2c")]
    use crate::metal::abstraction::limbs_conversion::h2c::{H2Fq, H2Fr, H2G};
    use crate::metal::tests::init_logger;
    #[cfg(feature = "ark")]
    use ark_ec::{AffineRepr, CurveGroup};
    #[cfg(feature = "ark")]
    use ark_ff::Field;
    #[cfg(feature = "ark")]
    use ark_std::UniformRand;
    #[cfg(feature = "h2c")]
    use halo2curves::{
        ff::Field as H2Field,
        group::prime::PrimeCurveAffine,
        group::{Curve, Group},
    };
    use proptest::arbitrary::any;
    use proptest::prelude::{prop, ProptestConfig};
    use proptest::{prop_assert_eq, prop_compose, proptest};
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use std::ops::Mul;

    #[cfg(feature = "h2c")]
    prop_compose! {
        fn rand_h2c_fq_element()(seed in any::<u64>()) -> H2Fq {
            let mut rng = StdRng::seed_from_u64(seed);
            H2Fq::random(&mut rng)
        }
    }

    #[cfg(feature = "h2c")]
    prop_compose! {
        fn rand_h2c_fr_element()(seed in any::<u64>()) -> H2Fr {
            let mut rng = StdRng::seed_from_u64(seed);
            H2Fr::random(&mut rng)
        }
    }

    #[cfg(feature = "h2c")]
    prop_compose! {
        fn rand_h2c_point()(seed in any::<u64>()) -> H2G {
            let mut rng = StdRng::seed_from_u64(seed);
            H2G::random(&mut rng)
        }
    }

    #[cfg(feature = "ark")]
    prop_compose! {
        fn rand_ark_fq_element()(seed in any::<u64>()) -> ArkFq {
            let mut rng = StdRng::seed_from_u64(seed);
            ArkFq::rand(&mut rng)
        }
    }

    #[cfg(feature = "ark")]
    prop_compose! {
        fn rand_ark_fr_element()(seed in any::<u64>()) -> ArkFr {
            let mut rng = StdRng::seed_from_u64(seed);
            ArkFr::rand(&mut rng)
        }
    }

    #[cfg(feature = "ark")]
    prop_compose! {
        fn rand_ark_point()(seed in any::<u64>()) -> ArkG {
            let mut rng = StdRng::seed_from_u64(seed);
            ArkG::rand(&mut rng)
        }
    }

    proptest! {

        #![proptest_config(ProptestConfig::with_cases(100))]

        #[cfg(feature = "h2c")]
        #[test]
        fn test_fr_serialisation_to_limbs_h2c(f in rand_h2c_fr_element()) {
            let limbs = f.to_u32_limbs();
            let f_prime = H2Fr::from_u32_limbs(&limbs);
            prop_assert_eq!(f, f_prime);
        }

        #[cfg(all(feature = "h2c", feature = "ark"))]
        #[test]
        fn test_fr_conversion_between_h2c_to_ark(f_h2c in rand_h2c_fr_element(), p in any::<u64>()) {
            let limbs = f_h2c.to_u32_limbs();
            let f_ark = ArkFr::from_u32_limbs(&limbs);

            prop_assert_eq!(f_h2c.to_u32_limbs(), f_ark.to_u32_limbs());

            let f_pow_h2c = f_h2c.pow(&[p]);
            let f_pow_ark = f_ark.pow(&[p]);

            prop_assert_eq!(f_pow_h2c.to_u32_limbs(), f_pow_ark.to_u32_limbs());
        }

        #[cfg(feature = "h2c")]
        #[test]
        fn test_fq_serialisation_to_limbs_h2c(f in rand_h2c_fq_element()) {
            let limbs = f.to_u32_limbs();
            let f_prime = H2Fq::from_u32_limbs(&limbs);
            prop_assert_eq!(f, f_prime);
        }

        #[cfg(feature = "ark")]
        #[test]
        fn test_fq_serialisation_to_limbs_ark(f in rand_ark_fq_element()) {
            let limbs = f.to_u32_limbs();
            let f_prime = ArkFq::from_u32_limbs(&limbs);
            prop_assert_eq!(f, f_prime);
        }

        #[cfg(feature = "ark")]
        #[test]
        fn test_list_serialisation_to_limbs_ark(fs in prop::collection::vec(rand_ark_fq_element(), 1..100)) {
            init_logger();

            // Convert the list of Fq elements to limbs using the &[P] implementation
            let list_limbs = fs.as_slice().to_u32_limbs();

            // Manually serialize each element and collect the limbs
            let manual_limbs: Vec<u32> = fs
                .iter()
                .flat_map(|f| f.to_u32_limbs())
                .collect();

            // Check that the serialized outputs match
            prop_assert_eq!(list_limbs.clone(), manual_limbs);

            // Deserialize the limbs back into Fq elements
            let fs_prime: Vec<ArkFq> = list_limbs
                .chunks_exact(8) // Assuming ArkFq has 8 limbs
                .map(ArkFq::from_u32_limbs)
                .collect();

            // Check that the original list and the deserialized list are equal
            prop_assert_eq!(fs, fs_prime);
        }

        #[cfg(feature = "h2c")]
        #[test]
        fn test_list_serialisation_to_limbs_h2c(fs in prop::collection::vec(rand_h2c_fq_element(), 1..100)) {
            init_logger();
            // Convert the list of Fq elements to limbs using the &[P] implementation
            let list_limbs = fs.as_slice().to_u32_limbs();

            // Manually serialize each element and collect the limbs
            let manual_limbs: Vec<u32> = fs
                .iter()
                .flat_map(|f| f.to_u32_limbs())
                .collect();

            // Check that the serialized outputs match
            prop_assert_eq!(list_limbs.clone(), manual_limbs);

            // Deserialize the limbs back into Fq elements
            let fs_prime: Vec<H2Fq> = list_limbs
                .chunks_exact(8) // Assuming ArkFq has 8 limbs
                .map(H2Fq::from_u32_limbs)
                .collect();

            // Check that the original list and the deserialized list are equal
            prop_assert_eq!(fs, fs_prime);
        }

        #[cfg(all(feature = "h2c", feature = "ark"))]
        #[test]
        fn test_fq_conversion_between_h2c_to_ark(f_h2c in rand_h2c_fq_element(), p in any::<u64>()) {
            let limbs = f_h2c.to_u32_limbs();
            let f_ark = ArkFq::from_u32_limbs(&limbs);

            prop_assert_eq!(f_h2c.to_u32_limbs(), f_ark.to_u32_limbs());

            let f_pow_h2c = f_h2c.pow(&[p]);
            let f_pow_ark = f_ark.pow(&[p]);

            prop_assert_eq!(f_pow_h2c.to_u32_limbs(), f_pow_ark.to_u32_limbs());
        }

        #[cfg(feature = "h2c")]
        #[test]
        fn test_point_serialisation_to_limbs_h2c(p in rand_h2c_point()) {
            let limbs = p.to_u32_limbs();
            println!("{:?}", limbs);
            let p_prime = H2G::from_u32_limbs(&limbs);
            prop_assert_eq!(p, p_prime);
        }

        #[cfg(all(feature = "h2c", feature = "ark"))]
        #[test]
        fn test_point_conversion_between_h2c_to_ark_canonical(p in rand_h2c_point(), m in rand_h2c_fr_element()) {
            // Convert point to canonical form
            let p_h2c = p.to_affine().to_curve();

            let p_ark = ArkG::from_u32_limbs(&p_h2c.to_u32_limbs());

            prop_assert_eq!(
                p_h2c.to_affine().to_curve().to_u32_limbs(),
                p_ark.into_affine().into_group().to_u32_limbs()
            );

            let p_mul_h2c = p_h2c.mul(m);
            let m_ark = ArkFr::from_u32_limbs(&m.to_u32_limbs());
            let p_mul_ark = p_ark.mul(m_ark);


            prop_assert_eq!(p_mul_h2c.to_affine().x.to_u32_limbs(), p_mul_ark.into_affine().x.to_u32_limbs());
            prop_assert_eq!(p_mul_h2c.to_affine().y.to_u32_limbs(), p_mul_ark.into_affine().y.to_u32_limbs());

            prop_assert_eq!(
                p_mul_h2c.to_affine().to_curve().to_u32_limbs(),
                p_mul_ark.into_affine().into_group().to_u32_limbs()
            );
        }

        #[cfg(all(feature = "h2c", feature = "ark"))]
        #[test]
        fn test_point_conversion_between_ark_to_h2c_canonical(p in rand_ark_point(), m in rand_ark_fr_element()) {
            // Convert point to canonical form
            let p_ark = p.into_affine().into_group();

            let p_h2c = H2G::from_u32_limbs(&p_ark.to_u32_limbs());

            prop_assert_eq!(
                p_h2c.to_affine().to_curve().to_u32_limbs(),
                p_ark.into_affine().into_group().to_u32_limbs()
            );

            let p_mul_ark = p_ark.mul(m);
            let m_h2c = H2Fr::from_u32_limbs(&m.to_u32_limbs());
            let p_mul_h2c = p_h2c.mul(m_h2c);

            prop_assert_eq!(p_mul_h2c.to_affine().x.to_u32_limbs(), p_mul_ark.into_affine().x.to_u32_limbs());
            prop_assert_eq!(p_mul_h2c.to_affine().y.to_u32_limbs(), p_mul_ark.into_affine().y.to_u32_limbs());

            prop_assert_eq!(
                p_mul_h2c.to_affine().to_curve().to_u32_limbs(),
                p_mul_ark.into_affine().into_group().to_u32_limbs()
            );
        }

    }
}
