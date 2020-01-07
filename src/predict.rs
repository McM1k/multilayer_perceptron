pub fn cross_entropy(y: Vec<_>, p: Vec<_>) -> f64 {
    let n = y.len();
    let mut sum = 0.0;

    for i in 0..n {
        sum += y[i] * p[i].log10() + (1 - y[i]) * (1 - p[i]).log10();
    }

    -sum / n as f64
}
