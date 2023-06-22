use std::path::Path;

use rust_ml::PreprocessService;
use rust_ml::TrainService;

const _DATA_PATH: &str = "./samples/input/penguins_size.csv";

fn main() {
    let path = Path::new(_DATA_PATH);
    let preprocess_service = PreprocessService::new(&path);
    let (features, target) = preprocess_service.create_feature_and_target_tables();
    let (x_train, x_test, y_train, y_test) =
        preprocess_service.split_train_and_test(features, target, 0.3);

    let train_service = TrainService::new(x_train, y_train, x_test, y_test, 5, 1, 2, 100);
    let random_forest_classifier = train_service.fit();
    train_service.validate(random_forest_classifier);
}
