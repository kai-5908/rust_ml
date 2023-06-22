use ::polars::frame::DataFrame;
use ::polars::prelude::Schema;
use ::polars::prelude::*;

use crate::validate;

pub struct SizeDataSchema {
    pub schema: Schema,
}

impl SizeDataSchema {
    pub fn new() -> Self {
        let mut schema = Schema::new();
        schema.with_column("species".to_string().into(), DataType::Utf8);
        schema.with_column("island".to_string().into(), DataType::Utf8);
        schema.with_column("culmen_length_mm".to_string().into(), DataType::Float64);
        schema.with_column("culmen_depth_mm".to_string().into(), DataType::Float64);
        schema.with_column("flipper_length_mm".to_string().into(), DataType::Float64);
        schema.with_column("body_mass_g".to_string().into(), DataType::Float64);
        schema.with_column("sex".to_string().into(), DataType::Utf8);
        SizeDataSchema { schema }
    }

    pub fn check(size_data: &DataFrame) {
        let col_names = vec![
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ];
        for col_name in col_names {
            let series: &Series = &size_data[col_name];
            match col_name {
                "culmen_length_mm" | "culmen_depth_mm" | "flipper_length_mm" | "body_mass_g" => {
                    validate::validate_positive_float_from_series(series)
                }
                "sex" => validate::validate_sex_from_series(series),
                _ => (),
            }
        }
    }

    pub fn species_str_to_num(str_val: &Series) -> Series {
        str_val
            .utf8()
            .unwrap()
            .into_iter()
            .map(|opt_name: Option<&str>| {
                opt_name.map(|name: &str| match name {
                    "Adelie" => 1,
                    "Chinstrap" | "Gentoo" => 0,
                    _ => panic!("Problem species str to num"),
                })
            })
            .collect::<UInt32Chunked>()
            .into_series()
    }
}
