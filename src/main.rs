use smartcore::dataset::iris;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::accuracy;

fn main() {
    let iris_data = iris::load_dataset();
    let x = DenseMatrix::from_array(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );
    let y = iris_data.target;

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

    let model = LogisticRegression::fit(
        &x_train,
        &y_train,
        Default::default(),
    ).unwrap();

    let pred = model.predict(&x_test).unwrap();
    println!("{:#?}", iris_data.num_features);
    println!("{}", accuracy(&y_test, &pred));
}