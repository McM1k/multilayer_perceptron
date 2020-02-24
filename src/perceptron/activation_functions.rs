use std::vec::Vec;

#[derive(Clone, Copy)]
pub enum ActivationNames {
    Sigmoid,
    HyperboloidTangent,
    RectifiedLinearUnit,
}

impl ActivationNames {
    pub fn get_fn(self) -> fn(f64) -> f64 {
        match self {
            ActivationNames::Sigmoid => sigmoid,
            ActivationNames::HyperboloidTangent => tanh,
            ActivationNames::RectifiedLinearUnit => relu,
        }
    }

    pub fn get_deriv(self) -> fn(f64) -> f64 {
        match self {
            ActivationNames::Sigmoid => sigmoid_deriv,
            ActivationNames::HyperboloidTangent => tanh_deriv,
            ActivationNames::RectifiedLinearUnit => relu_deriv,
        }
    }
}

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub fn sigmoid_deriv(z: f64) -> f64 {
    let sigz = sigmoid(z);
    sigz * (1.0 - sigz)
}

pub fn tanh(z: f64) -> f64 {
    (z.exp() - (-z).exp()) / (z.exp() + (-z).exp())
}

pub fn tanh_deriv(z: f64) -> f64 {
    let coshz = (z.exp() + (-z).exp()) / 2.0;
    let sinhz = (z.exp() - (-z).exp()) / 2.0;
    ((coshz * coshz) - (sinhz * sinhz)) / (coshz * coshz)
}

pub fn relu(z: f64) -> f64 {
    if z <= 0.0 {
        0.0
    } else {
        z
    }
}

pub fn relu_deriv(z: f64) -> f64 {
    if z <= 0.0 {
        0.0
    } else {
        1.0
    }
}

pub fn softmax(x: &[f64]) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::new();

    for xi in x.iter() {
        let mut sum = 0.0;

        for xj in x.iter() {
            sum += xj.exp();
        }

        vec.push(xi.exp() / sum);
    }

    vec
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

            assert_eq!(softmax(&x), [1.0]);
        }
    }
}
