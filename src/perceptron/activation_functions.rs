use std::vec::Vec;

pub fn sigmoid(z: &f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub fn tanh(z: &f64) -> f64 {
    (z.exp() - (-z).exp()) / (z.exp() + (-z).exp())
}

pub fn relu(z: &f64) -> f64 {
    if *z <= 0.0 {
        0.0
    }
    else {
        *z
    }
}

pub fn softmax(x: &Vec<f64>, i: usize) -> f64 {
    let mut sum = 0.0;

    for xj in x.iter() {
        sum += xj.exp();
    }

    x[i].exp() / sum
}



#[cfg(test)]
mod activation_functions_tests {
    mod sigmoid {
        use super::super::sigmoid;

        #[test]
        fn z_nul() {
            let z = 0.0;
            assert_eq!(sigmoid(&z), 0.5);
        }

        #[test]
        fn z_neg() {
            let z = -1.0;
            let result = sigmoid(&z);
            assert!(result < 0.5 && result >= 0.0);
        }

        #[test]
        fn z_pos() {
            let z = 1.0;
            let result = sigmoid(&z);
            assert!(result > 0.5 && result <= 1.0);
        }
    }

    mod tanh {
        use super::super::tanh;

        #[test]
        fn z_nul() {
            let z = 0.0;
            assert_eq!(tanh(&z), 0.0);
        }

        #[test]
        fn z_neg() {
            let z = -1.0;
            let result = tanh(&z);
            assert!(result < 0.0 && result >= -1.0);
        }

        #[test]
        fn z_pos() {
            let z = 1.0;
            let result = tanh(&z);
            assert!(result > 0.0 && result <= 1.0);
        }
    }

    mod relu {
        use super::super::relu;

        #[test]
        fn z_neg() {
            let z = -1.0;
            assert_eq!(relu(&z), 0.0);
        }

        #[test]
        fn z_pos() {
            let z = 1.0;
            assert_eq!(relu(&z), z);
        }
    }

    mod softmax {
        use super::super::softmax;

        #[test]
        fn unit_vec() {
            let x = vec![42.0];

            assert_eq!(softmax(&x, 0), 1.0);
        }
    }
}