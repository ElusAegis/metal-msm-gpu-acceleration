use rand::RngCore;

// implement to_u32_limbs and from_u32_limbs for BigInt<4>
pub trait ToLimbs {
    fn to_u32_limbs(&self) -> Vec<u32>;
}

pub trait FromLimbs {
    const U32_SIZE : usize;

    fn from_u32_limbs(limbs: &[u32]) -> Self;
}

pub trait ScalarGPU : ToLimbs {
    const MODULUS_BIT_SIZE : usize;

    fn random(rng: &mut impl RngCore) -> Self;

    fn into<B : ScalarGPU + FromLimbs>(&self) -> B {
        assert_eq!(Self::MODULUS_BIT_SIZE, B::MODULUS_BIT_SIZE, "Incompatible scalar sizes");
        B::from_u32_limbs(&self.to_u32_limbs())
    }
}

pub trait PointGPU : FromLimbs + ToLimbs {
    fn random(rng: &mut impl RngCore) -> Self;

    fn into<B : PointGPU>(&self) -> B {
        assert_eq!(Self::U32_SIZE, B::U32_SIZE, "Incompatible point sizes");
        B::from_u32_limbs(&self.to_u32_limbs())
    }
}

#[cfg(feature = "ark")]
pub mod ark {
    use ark_ff::{BigInteger, BigInteger256, PrimeField};
    use super::{FromLimbs, PointGPU, ScalarGPU, ToLimbs};

    pub use ark_bn254::{Fq as ArkFq, G1Projective as ArkG, Fr as ArkFr, G1Affine as ArkGAffine};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_std::UniformRand;
    use rand::RngCore;

    // convert from little endian to big endian
    impl ToLimbs for BigInteger256 {
        fn to_u32_limbs(&self) -> Vec<u32> {
            let mut limbs = Vec::new();
            self.to_bytes_be().chunks(8).for_each(|chunk| {
                let high = u32::from_be_bytes(chunk[0..4].try_into().unwrap());
                let low = u32::from_be_bytes(chunk[4..8].try_into().unwrap());
                limbs.push(high);
                limbs.push(low);
            });
            limbs
        }
    }

    // convert from little endian to big endian
    impl ToLimbs for ArkFq {
        fn to_u32_limbs(&self) -> Vec<u32> {
            self.0.to_u32_limbs()
        }
    }

    impl ToLimbs for ArkFr {
        fn to_u32_limbs(&self) -> Vec<u32> {
            self.into_bigint().to_u32_limbs()
        }
    }

    impl ToLimbs for ArkG {
        fn to_u32_limbs(&self) -> Vec<u32> {
             self.x.to_u32_limbs().into_iter()
                .chain(self.y.to_u32_limbs())
                .chain(self.z.to_u32_limbs())
                .collect::<Vec<_>>()
        }
    }

    impl ToLimbs for ArkGAffine {
        fn to_u32_limbs(&self) -> Vec<u32> {
            self.into_group().to_u32_limbs()
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
            let a = ArkFq::new_unchecked(BigInteger256::from_u32_limbs(limbs));
            a
        }
    }

    impl FromLimbs for ArkFr {
        const U32_SIZE: usize = 8;

        fn from_u32_limbs(limbs: &[u32]) -> Self {
            Self::from(BigInteger256::from_u32_limbs(limbs))
        }
    }

    impl ScalarGPU for ArkFr {
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

    impl PointGPU for ArkG {
        fn random(rng: &mut impl RngCore) -> Self {
            <Self as UniformRand>::rand(rng).into_affine().into_group()
        }
    }
}

#[cfg(feature = "h2c")]
pub mod h2c {
    use super::{FromLimbs, PointGPU, ScalarGPU, ToLimbs};
    use halo2curves::CurveExt;

    pub use halo2curves::bn256::{Fq as H2Fq, Fr as H2Fr, G1 as H2G, G1Affine as H2GAffine};
    use halo2curves::ff::{Field};
    use halo2curves::group::{Curve, Group};
    use halo2curves::group::prime::PrimeCurveAffine;
    use halo2curves::serde::SerdeObject;
    use rand::RngCore;

    impl ToLimbs for H2Fr {
        fn to_u32_limbs(&self) -> Vec<u32> {
            let mut output = [0u32; 8];
            let input = self.to_bytes();
            for (i, chunk) in input.chunks_exact(4).rev().enumerate() {
                output[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            output.to_vec()
        }
    }

    impl ToLimbs for H2Fq {
        fn to_u32_limbs(&self) -> Vec<u32> {
            let mut output = [0u32; 8];
            let input = self.to_raw_bytes();
            for (i, chunk) in input.chunks_exact(4).rev().enumerate() {
                output[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            output.to_vec()
        }
    }


    impl ToLimbs for H2G {
        fn to_u32_limbs(&self) -> Vec<u32> {
            self.x.to_u32_limbs()
                .into_iter()
                .chain(self.y.to_u32_limbs())
                .chain(self.z.to_u32_limbs())
                .collect::<Vec<_>>()
        }
    }

    impl ToLimbs for H2GAffine {
        fn to_u32_limbs(&self) -> Vec<u32> {
            self.to_curve().to_u32_limbs()
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

    impl ScalarGPU for H2Fr {
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
            ).expect("Failed to create `new_jacobian` point from limbs")
        }
    }

    impl PointGPU for H2G {
        fn random(rng: &mut impl RngCore) -> Self {
            <Self as Group>::random(rng).to_affine().to_curve()
        }
    }
}

#[cfg(all(test, feature = "h2c", feature = "ark"))]
mod test {
    use std::ops::Mul;
    use ark_ff::{Field as ArkField};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_std::UniformRand;
    use halo2curves::ff::{Field as H2Field};
    use halo2curves::group::{Curve, Group};
    use halo2curves::group::prime::PrimeCurveAffine;
    use proptest::{prop_assert_eq, prop_compose, proptest};
    use proptest::arbitrary::any;
    use proptest::prelude::ProptestConfig;
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use crate::msm::metal::abstraction::limbs_conversion::ark::{ArkFq, ArkFr, ArkG};
    use crate::msm::metal::abstraction::limbs_conversion::h2c::{H2Fq, H2Fr, H2G};
    use super::{FromLimbs, ToLimbs};

    prop_compose! {
        fn rand_h2c_fq_element()(seed in any::<u64>()) -> H2Fq {
            let mut rng = StdRng::seed_from_u64(seed);
            H2Fq::random(&mut rng)
        }
    }

    prop_compose! {
        fn rand_ark_fq_element()(seed in any::<u64>()) -> ArkFq {
            let mut rng = StdRng::seed_from_u64(seed);
            ArkFq::rand(&mut rng)
        }
    }

    prop_compose! {
        fn rand_h2c_fr_element()(seed in any::<u64>()) -> H2Fr {
            let mut rng = StdRng::seed_from_u64(seed);
            H2Fr::random(&mut rng)
        }
    }

    prop_compose! {
        fn rand_h2c_point()(seed in any::<u64>()) -> H2G {
            let mut rng = StdRng::seed_from_u64(seed);
            H2G::random(&mut rng)
        }
    }

    prop_compose! {
        fn rand_ark_fr_element()(seed in any::<u64>()) -> ArkFr {
            let mut rng = StdRng::seed_from_u64(seed);
            ArkFr::rand(&mut rng)
        }
    }

    prop_compose! {
        fn rand_ark_point()(seed in any::<u64>()) -> ArkG {
            let mut rng = StdRng::seed_from_u64(seed);
            ArkG::rand(&mut rng)
        }
    }

proptest! {

        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_fr_serialisation_to_limbs_h2c(f in rand_h2c_fr_element()) {
            let limbs = f.to_u32_limbs();
            let f_prime = H2Fr::from_u32_limbs(&limbs);
            prop_assert_eq!(f, f_prime);
        }

        #[test]
        fn test_fr_conversion_between_h2c_to_ark(f_h2c in rand_h2c_fr_element(), p in any::<u64>()) {
            let limbs = f_h2c.to_u32_limbs();
            let f_ark = ArkFr::from_u32_limbs(&limbs);

            prop_assert_eq!(f_h2c.to_u32_limbs(), f_ark.to_u32_limbs());

            let f_pow_h2c = f_h2c.pow(&[p]);
            let f_pow_ark = f_ark.pow(&[p]);

            prop_assert_eq!(f_pow_h2c.to_u32_limbs(), f_pow_ark.to_u32_limbs());
        }

        #[test]
        fn test_fq_serialisation_to_limbs_h2c(f in rand_h2c_fq_element()) {
            let limbs = f.to_u32_limbs();
            let f_prime = H2Fq::from_u32_limbs(&limbs);
            prop_assert_eq!(f, f_prime);
        }

        #[test]
        fn test_fq_serialisation_to_limbs_ark(f in rand_ark_fq_element()) {
            let limbs = f.to_u32_limbs();
            let f_prime = ArkFq::from_u32_limbs(&limbs);
            prop_assert_eq!(f, f_prime);
        }

        #[test]
        fn test_fq_conversion_between_h2c_to_ark(f_h2c in rand_h2c_fq_element(), p in any::<u64>()) {
            let limbs = f_h2c.to_u32_limbs();
            let f_ark = ArkFq::from_u32_limbs(&limbs);

            prop_assert_eq!(f_h2c.to_u32_limbs(), f_ark.to_u32_limbs());

            let f_pow_h2c = f_h2c.pow(&[p]);
            let f_pow_ark = f_ark.pow(&[p]);

            prop_assert_eq!(f_pow_h2c.to_u32_limbs(), f_pow_ark.to_u32_limbs());
        }

        #[test]
        fn test_point_serialisation_to_limbs_h2c(p in rand_h2c_point()) {
            let limbs = p.to_u32_limbs();
            let p_prime = H2G::from_u32_limbs(&limbs);
            prop_assert_eq!(p, p_prime);
        }

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