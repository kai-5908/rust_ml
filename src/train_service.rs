use smartcore::ensemble::random_forest_classifier::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::metrics::ClassificationMetrics;

pub struct TrainService {
    train_x: DenseMatrix<f64>,
    train_y: Vec<f64>,
    valid_x: DenseMatrix<f64>,
    valid_y: Vec<f64>,
    max_depth: u16,
    min_samples_leaf: usize,
    min_samples_split: usize,
    n_trees: u16,
}

impl TrainService {
    pub fn new(
        train_x: DenseMatrix<f64>,
        train_y: Vec<f64>,
        valid_x: DenseMatrix<f64>,
        valid_y: Vec<f64>,
        max_depth: u16,
        min_samples_leaf: usize,
        min_samples_split: usize,
        n_trees: u16,
    ) -> Self {
        TrainService {
            train_x,
            train_y,
            valid_x,
            valid_y,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            n_trees,
        }
    }

    pub fn fit(&self) -> RandomForestClassifier<f64> {
        let mut parameters = RandomForestClassifierParameters::default();
        parameters = parameters.with_max_depth(self.max_depth);
        parameters = parameters.with_min_samples_leaf(self.min_samples_leaf);
        parameters = parameters.with_min_samples_split(self.min_samples_split);
        parameters = parameters.with_n_trees(self.n_trees);
        let classifier =
            RandomForestClassifier::fit(&self.train_x, &self.train_y, parameters).unwrap();
        classifier
    }

    pub fn validate(&self, classifier: RandomForestClassifier<f64>) {
        let predict_y = classifier.predict(&self.valid_x).unwrap();
        let accuracy = ClassificationMetrics::accuracy().get_score(&self.valid_y, &predict_y);
        let recall = ClassificationMetrics::recall().get_score(&self.valid_y, &predict_y);
        let precision = ClassificationMetrics::precision().get_score(&self.valid_y, &predict_y);
        let f1 = ClassificationMetrics::f1(0.5).get_score(&self.valid_y, &predict_y);
        let roc_auc = ClassificationMetrics::roc_auc_score().get_score(&self.valid_y, &predict_y);

        println!(
            "accuracy: {}, recall: {}, precision: {}, f1: {}, roc_auc: {}",
            accuracy, recall, precision, f1, roc_auc
        );
        for (predict, valid) in predict_y.iter().zip(self.valid_y.iter()) {
            println!("predict: {}, actual: {}", predict, valid);
        }
    }
}
