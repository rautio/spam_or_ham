use rustlearn::cross_validation::CrossValidation;
use rustlearn::feature_extraction::DictVectorizer;
use rustlearn::linear_models::sgdclassifier;
use rustlearn::metrics::accuracy_score;
use rustlearn::prelude::*;
use std::io::Read;
use std::time::Instant;

// Convert input data to feature matrix
fn parse(data: &str) -> (SparseRowArray, Array) {
    // Initialize the vectorizer
    let mut vectorizer = DictVectorizer::new();
    let mut labels = Vec::new();

    for (row_num, line) in data.lines().enumerate() {
        // Split each line to a label ("ham" or "spam") and the content of the SMS message
        let (label, text) = line.split_at(line.find('\t').unwrap());
        // Convert the label to binary
        labels.push(match label {
            "spam" => 0.0,
            "ham" => 1.0,
            _ => panic!("{}", format!("Invalid label: {}", label)),
        });

        // Convert the SMS message to tokens
        for token in text.split_whitespace() {
            vectorizer.partial_fit(row_num, token, 1.0);
        }
    }

    (vectorizer.transform(), Array::from(labels))
}

// Download remote data using a url
fn download_data(url: &str) -> String {
    let mut tmpfile = tempfile::tempfile().unwrap();
    let _ = reqwest::blocking::get(url).unwrap().copy_to(&mut tmpfile);
    let mut zip = zip::ZipArchive::new(tmpfile).unwrap();
    let mut file_zip = zip.by_name("SMSSpamCollection").unwrap();

    let mut data = String::new();
    file_zip.read_to_string(&mut data).unwrap();

    data
}

fn fit(x: &SparseRowArray, y: &Array) -> (f32, f32) {
    let num_epochs = 10;
    let num_folds = 10;

    let mut test_accuracy = 0.0;
    let mut train_accuracy = 0.0;

    for (train_indicies, test_indices) in CrossValidation::new(y.rows(), num_folds) {
        let x_train = x.get_rows(&train_indicies);
        let x_test = x.get_rows(&test_indices);

        let y_train = y.get_rows(&train_indicies);
        let y_test = y.get_rows(&test_indices);

        let mut model = sgdclassifier::Hyperparameters::new(x.cols())
            .learning_rate(0.05)
            .l2_penalty(0.01)
            .build();

        for _ in 0..num_epochs {
            model.fit(&x_train, &y_train).unwrap();
        }

        let fold_test_accuracy = accuracy_score(&y_test, &model.predict(&x_test).unwrap());
        let fold_train_accuracy = accuracy_score(&y_train, &model.predict(&x_train).unwrap());

        test_accuracy += fold_test_accuracy;
        train_accuracy += fold_train_accuracy;
    }

    (
        test_accuracy / num_folds as f32,
        train_accuracy / num_folds as f32,
    )
}

fn main() {
    let data = download_data(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    );
    let (x, y) = parse(&data);
    println!(
        "X: {} rows, {} columns, {} non-zero entries, Y: {:.2}% positive class",
        x.rows(),
        x.cols(),
        x.nnz(),
        y.mean() * 100.0
    );
    let now = Instant::now();
    // let start_time = precise_time_ns();
    let (test_accuracy, train_accuracy) = fit(&x, &y);
    // let duration = precise_time_ns() - start_time;
    println!("Training time: {:?}", now.elapsed());
    println!(
        "Test accuracy: {:.3}, train accuracy: {:.3}",
        test_accuracy, train_accuracy
    );
}
