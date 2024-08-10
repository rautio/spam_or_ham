fn main() {
    let zipped = download_data(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    );
    println!("Downloaded {} bytes of data", zipped.len())
}

fn download_data(url: &str) -> Vec<u8> {
    let response = reqwest::blocking::get(url).unwrap().text().unwrap();
    response.into_bytes()
}
