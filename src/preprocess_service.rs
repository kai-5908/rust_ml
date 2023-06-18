use std::convert::TryFrom;
use std::fs::File;
use std::path::Path;

use polars::frame::DataFrame;
use polars::prelude::SerReader;
use polars::prelude::*;

use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;

use crate::input_schema;

struct PreprocessService {
    size_data: DataFrame,
}

impl PreprocessService {
    pub fn new(path: &Path) -> Self {
        let file = File::open(path).expect("cannot open file");
        let schema = input_schema::SizeDataSchema::new().schema;
        let size_data_with_null = CsvReader::new(file)
            .with_schema(Arc::new(schema))
            .has_header(true)
            .finish();
        let size_data = size_data_with_null
            .expect("not validated")
            .drop_nulls::<String>(None)
            .unwrap();
        input_schema::SizeDataSchema::check(size_data);
        PreprocessService { size_data }
    }

    pub fn create_feature_and_target_tables(&self) -> (Result<DataFrame, PolarsError>, Vec<f64>) {
        let features = self.size_data.select(vec![
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]);
        let species = self.size_data.select(vec!["species"]);
        let target_array = species
            .unwrap()
            .apply("species", input_schema::SizeDataSchema::species_str_to_num)
            .unwrap()
            .to_ndarray::<Float64Type>()
            .unwrap();
        let mut target: Vec<f64> = Vec::new();
        for val in target_array.iter() {
            target.push(*val);
        }
        (features, target)
    }
}

pub fn convert_features_to_matrix(size_data: &DataFrame) -> Result<DenseMatrix<f64>, PolarsError> {
    let nrows = size_data.height();
    let ncols = size_data.width();

    let features_res = size_data.to_ndarray::<Float64Type>().unwrap();
    let mut xmatrix: DenseMatrix<f64> = BaseMatrix::zeros(nrows, ncols);
    let mut col: u32 = 0;
    let mut row: u32 = 0;

    for val in features_res.iter() {
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        xmatrix.set(m_row, m_col, *val);
        if m_col == ncols - 1 {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    Ok(xmatrix)
}
