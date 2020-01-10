pub fn cross_entropy(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len();
    let mut sum = 0.0;

    for i in 0..n {
        sum += y[i] * p[i].log10() + (1.0 - y[i]) * (1.0 - p[i]).log10();
    }

    -sum / n as f64
}
