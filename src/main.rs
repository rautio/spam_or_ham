use rustlearn::feature_extraction::DictVectorizer;
use rustlearn::prelude::*;
use std::io::Read;

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
}
