use std::io::{Cursor, Read};
use zip::read::ZipArchive;

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
    println!("Downloaded {} lines", data.len());

    for line in data.lines().take(3) {
        println!("{}", line);
    }
}
