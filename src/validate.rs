use polars::series::Series;

pub fn validate_positive_float_from_series(float_series: &Series) {
    let series_iter = float_series
        .f64()
        .expect("series was not an f64 dtype")
        .into_iter();
    for val in series_iter {
        match val {
            Some(val) if val > 0.0 => (),
            _ => panic!("value is negative or null"),
        }
    }
}
pub fn validate_sex_from_series(str_series: &Series) {
    let series_iter = str_series
        .utf8()
        .expect("series was not an utf8 dtype")
        .into_iter();
    for val in series_iter {
        match val {
            Some(val) if val == "MALE" || val == "FEMALE" => (),
            _ => panic!("sex str value is not valid"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::validate::*;
    use polars::series::Series;
    #[test]
    fn test_validate_positive_float_from_series_normal() {
        let sample_series: Series = [11.1, 22.2, 33.3].iter().collect();
        validate_positive_float_from_series(&sample_series)
    }

    #[test]
    #[should_panic]
    fn test_validate_positive_float_from_series_anomaly() {
        let sample_series: Series = [11.1, 22.2, -33.3].iter().collect();
        validate_positive_float_from_series(&sample_series)
    }
}
