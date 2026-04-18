#[derive(Debug)]
pub struct PartMin {
    min_vec: Vec<f64>,
}

impl PartMin {
    pub fn from_examples(
        sizes: impl IntoIterator<Item = usize>,
        values: impl IntoIterator<Item = f64>,
        capacity: usize,
    ) -> Self {
        let mut init_min_vec = Vec::new();
        for (size, value) in sizes.into_iter().zip(values) {
            if size == 0 {
                continue;
            }
            if size > init_min_vec.len() {
                init_min_vec.extend(std::iter::repeat_n(
                    f64::INFINITY,
                    size - init_min_vec.len(),
                ))
            }
            if value < init_min_vec[size - 1] {
                init_min_vec[size - 1] = value
            }
        }

        let mut final_part = PartMin { min_vec: vec![] };

        for n in 0..capacity {
            if n < init_min_vec.len() {
                final_part.extend_one(Some(init_min_vec[n]))
            } else {
                final_part.extend_one(None)
            }
        }

        final_part
    }

    pub fn extend_one(&mut self, new_val: Option<f64>) {
        let self_vec = &self.min_vec;
        let cur_len = self_vec.len();
        let mut base = match new_val {
            None => f64::INFINITY,
            Some(v) => v,
        };
        if cur_len == 0 {
            self.min_vec.push(base);
            return;
        }
        let vec1 = &self_vec[..cur_len.div_ceil(2)];
        let vec2 = &self_vec[cur_len / 2..];

        for (x1, x2) in vec1.iter().zip(vec2.iter().rev()) {
            let sum = x1 + x2;
            if sum < base {
                base = sum
            }
        }
        self.min_vec.push(base);
    }

    pub fn min_value(&self, size: usize) -> f64 {
        if size == 0 {
            0f64
        } else if size <= self.min_vec.len() {
            self.min_vec[size - 1]
        } else if let Some(&v) = self.min_vec.last() {
            v
        } else {
            0f64
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.min_vec.len()
    }
}

#[cfg(test)]
mod tests {
    use super::PartMin;

    #[test]
    fn test_empty() {
        let empty_part = PartMin::from_examples(vec![], vec![], 0);
        assert_eq!(empty_part.len(), 0);
        assert_eq!(empty_part.min_value(0), 0f64);
        assert_eq!(empty_part.min_value(1), 0f64);
    }

    #[test]
    fn test_zero_size() {
        let empty_part = PartMin::from_examples(vec![0], vec![1.], 0);
        assert_eq!(empty_part.len(), 0);
        assert_eq!(empty_part.min_value(0), 0f64);
        assert_eq!(empty_part.min_value(1), 0f64);
        let one_part = PartMin::from_examples(vec![1], vec![1.], 0);
        assert_eq!(one_part.len(), 0);
        assert_eq!(one_part.min_value(0), 0f64);
        assert_eq!(one_part.min_value(1), 0f64);
    }

    #[test]
    fn test_one() {
        let one_part = PartMin::from_examples(vec![1], vec![1.], 1);
        assert_eq!(one_part.len(), 1);
        assert_eq!(one_part.min_value(0), 0f64);
        assert_eq!(one_part.min_value(1), 1f64);
        assert_eq!(one_part.min_value(2), 1f64);
    }

    #[test]
    fn test_several() {
        let multi_part = PartMin::from_examples(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![9., 2., 5., 4., 6., 8., 10., 7., 5., 2.],
            14,
        );
        let target =
            vec![9., 2., 5., 4., 6., 6., 8., 7., 5., 2., 7., 4., 7., 6.];
        let result = Vec::from_iter((1..15).map(|x| multi_part.min_value(x)));
        assert_eq!(result, target);
    }

    #[test]
    fn test_float() {
        let multi_part = PartMin::from_examples(
            vec![1, 2],
            vec![2.909534676617408, 2.6994108385115014],
            14,
        );
        let target = vec![
            2.909534676617408,
            2.6994108385115014,
            5.60894551512891,
            5.398821677023003,
            8.308356353640411,
            8.098232515534505,
            11.007767192151913,
            10.797643354046006,
            13.707178030663414,
            13.497054192557506,
            16.406588869174914,
            16.196465031069007,
            19.105999707686415,
            18.895875869580507,
        ];
        let result = Vec::from_iter((1..15).map(|x| multi_part.min_value(x)));
        assert_eq!(result, target);
    }
}
