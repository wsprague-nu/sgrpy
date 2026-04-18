use itertools::Itertools;
use ordered_float::OrderedFloat;

/// Compute the log of the sum of exponentials of input elements.
pub fn logsumexp(a: Vec<f64>) -> f64 {
    if !a.is_empty() {
        _logsumexp(a)
    } else {
        f64::NEG_INFINITY
    }
}

fn _locate_max(a: &[f64]) -> (f64, Vec<bool>) {
    let real_a = a.iter().map(|&x| OrderedFloat(x)).max().unwrap().0;
    let mask = a.iter().map(|&x| x == real_a).collect();
    (real_a, mask)
}

/// Note: a must have length > 0
fn _logsumexp(mut a: Vec<f64>) -> f64 {
    // locate largest values
    let (a_max, i_max) = _locate_max(&a);

    // scale largest values and remove from sum (to be re-added later)
    for (a_entry, _) in a.iter_mut().zip(i_max.iter()).filter(|&(_, y)| *y) {
        *a_entry = f64::NEG_INFINITY;
    }

    let i_max_dt = i_max.into_iter().map(|x| if x { 1.0 } else { 0.0 });

    let m: f64 = i_max_dt.sum();

    // remove unused infinities
    let shift = if a_max.is_finite() { a_max } else { 0. };

    // perform logsumexp
    let exp = a.iter().map(|x| f64::exp(x - shift)).collect_vec();
    let mut s: f64 = exp.into_iter().sum();
    s = if s == 0. { s } else { s / m };

    s.ln_1p() + m.ln() + a_max
}

#[cfg(test)]
mod tests {
    use super::logsumexp;
    use approx::assert_relative_eq;

    fn check_rel_eq(a: f64, b: f64) {
        let rtol = 4. * f64::EPSILON.sqrt();
        assert_relative_eq!(a, b, max_relative = rtol);
    }

    #[test]
    fn test_logsumexp_1() {
        let a = vec![];
        let desired = f64::NEG_INFINITY;
        assert_eq!(logsumexp(a), desired);
    }

    #[test]
    fn test_logsumexp_2() {
        let a: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let desired = a.iter().map(|x| x.exp()).sum::<f64>().ln();

        check_rel_eq(logsumexp(a), desired);
    }

    #[test]
    fn test_logsumexp_3() {
        let a: Vec<f64> = vec![1000., 1000.];
        let desired = 1000. + f64::ln(2.);

        check_rel_eq(logsumexp(a), desired);
    }

    #[test]
    fn test_logsumexp_4() {
        let n = 1000;
        let a: Vec<f64> = vec![10000.; n];
        let desired = 10000. + f64::ln(n as f64);

        check_rel_eq(logsumexp(a), desired);
    }

    #[test]
    fn test_logsumexp_5() {
        let n = 1000000;
        let x: Vec<f64> = vec![1e-40; n];
        let logx: Vec<f64> = x.iter().map(|x| x.ln()).collect();

        check_rel_eq(logsumexp(logx).exp(), x.iter().sum());
    }

    #[test]
    fn test_logsumexp_6() {
        let inf = vec![f64::INFINITY];
        let neg_inf = vec![f64::NEG_INFINITY];
        let nan = vec![f64::NAN];

        assert_eq!(logsumexp(inf.clone()), inf[0]);
        assert_eq!(logsumexp(neg_inf.clone()), neg_inf[0]);
        assert!(logsumexp(nan.clone()).is_nan());
        assert_eq!(logsumexp(vec![neg_inf[0], neg_inf[0]]), neg_inf[0]);
    }

    #[test]
    fn test_logsumexp_7() {
        let a = vec![1e10, 1e-10];
        let desired = 1e10;

        check_rel_eq(logsumexp(a), desired);
    }

    #[test]
    fn test_logsumexp_8() {
        let a = vec![-1e10, f64::NEG_INFINITY];
        let desired = -1e10;

        check_rel_eq(logsumexp(a), desired);
    }
}
