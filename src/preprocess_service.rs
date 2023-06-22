use std::convert::TryFrom;
use std::fs::File;
use std::path::Path;

use polars::frame::DataFrame;
use polars::prelude::SerReader;
use polars::prelude::*;

use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::model_selection::train_test_split;

use crate::input_schema;

pub struct PreprocessService {
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
        input_schema::SizeDataSchema::check(&size_data);
        PreprocessService { size_data }
    }

    pub fn create_feature_and_target_tables(&self) -> (DenseMatrix<f64>, Vec<f64>) {
        let features_table = self.size_data.select(vec![
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]);
        let features = convert_features_to_matrix(&features_table.unwrap()).unwrap();
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

    pub fn split_train_and_test(&self, features:DenseMatrix<f64>, target: Vec<f64>, ratio: f32) -> (DenseMatrix<f64>, DenseMatrix<f64>,Vec<f64>, Vec<f64>){
        let (x_train, x_test, y_train, y_test) = train_test_split(&features, &target, ratio, true);
        (x_train, x_test, y_train, y_test)
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

#[cfg(test)]
mod tests {
    use crate::preprocess_service::PreprocessService;
    use polars::frame::DataFrame;
    use polars::prelude::*;
    use std::path::Path;

    #[test]
    fn test_new() {
        PreprocessService::new(Path::new("./samples/input/penguins_size.csv"));
    }

    #[test]
    fn test_create_feature_and_target_tables() {
        let s0 = Series::new("species", &["Adelie"]);
        let s1 = Series::new("island", &["Torgersen"]);
        let s2 = Series::new("culmen_length_mm", &[39.1]);
        let s3 = Series::new("culmen_depth_mm", &[18.7]);
        let s4 = Series::new("flipper_length_mm", &[181]);
        let s5 = Series::new("body_mass_g", &[3750]);
        let s6 = Series::new("sex", &["MALE"]);
        let size_data = DataFrame::new(vec![s0, s1, s2, s3, s4, s5, s6]).unwrap();

        let preprocess_service = PreprocessService { size_data };
        _ = preprocess_service.create_feature_and_target_tables()
    }
}
