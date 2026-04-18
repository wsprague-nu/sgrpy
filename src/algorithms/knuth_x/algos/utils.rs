use std::f64::consts::PI;

use statrs::function::gamma::ln_gamma;

/// ln(n!) function
fn lnfac_unchecked(n: usize) -> f64 {
    if n == 0 { 0. } else { ln_gamma((n + 1) as f64) }
}

/// ln((2n-1)!!) function (n should be positive nonzero)
fn lndfac_unchecked(n: usize) -> f64 {
    if n == 0 || n == 1 {
        0.
    } else {
        (n as f64) * (2f64).ln() - PI.ln() / 2. + ln_gamma(n as f64 + 0.5)
    }
}

pub fn calc_ac_factor(
    ccounts: impl IntoIterator<Item = usize>,
    v_remaining: usize,
) -> f64 {
    let ccounts_vec: Vec<usize> = ccounts.into_iter().collect();
    let ccounts_total = ccounts_vec.iter().sum::<usize>();

    let composition_cost =
        ccounts_vec.into_iter().map(lnfac_unchecked).sum::<f64>();
    let combination_cost = -lnfac_unchecked(ccounts_total + v_remaining);
    composition_cost + combination_cost
}

pub fn calc_as_factor(
    ecounts: impl IntoIterator<Item = usize>,
    etotal: usize,
) -> f64 {
    lndfac_unchecked(etotal - ecounts.into_iter().sum::<usize>())
}

// pub fn calc_entropy(weights: &[f64]) -> f64 {
//     let weight_sum: f64 = weights.iter().sum();
//     weights
//         .iter()
//         .map(|&w| w / weight_sum)
//         .map(|p: f64| -p * p.ln())
//         .sum()
// }

#[cfg(test)]
mod tests {

    use super::{lndfac_unchecked, lnfac_unchecked};
    use approx::assert_relative_eq;

    fn check_rel_eq(a: f64, b: f64) {
        let rtol = 4. * f64::EPSILON.sqrt();
        assert_relative_eq!(a, b, max_relative = rtol);
    }

    /// test lnfac
    #[test]
    fn test_lnfac() {
        assert_eq!(lnfac_unchecked(0), 0.);
        assert_eq!(lnfac_unchecked(1), 0.);
        check_rel_eq(lnfac_unchecked(2), (2f64).ln());
        check_rel_eq(lnfac_unchecked(3), (6f64).ln());
    }

    /// test lndfac
    #[test]
    fn test_lndfac() {
        assert_eq!(lndfac_unchecked(0), 0.);
        assert_eq!(lndfac_unchecked(1), 0.);
        check_rel_eq(lndfac_unchecked(2), 3f64.ln());
        check_rel_eq(lndfac_unchecked(3), 15f64.ln());
        check_rel_eq(lndfac_unchecked(8), 2027025f64.ln());
    }
}
