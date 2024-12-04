use kmedoids::labeledpam;
use ndarray::{Array1, Array2};
use std::env;
use std::error::Error;
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};


fn read_data(path:&str) -> Array2<f64>{
    let file = File::open(path).unwrap();
    let buffer = BufReader::new(file);
    let mut data = Vec::new();
    for line in buffer.lines() {
        let line = line.unwrap();
        let mut vec = Vec::new();
        for num in line.split(",") {
            vec.push(num.parse::<f64>().unwrap());
        }
        data.push(vec);
    };
    let mut dist_mat=Array2::<f64>::zeros((data.len(),data.len()));
    for i in 0..data.len(){
        for j in i..data.len(){
            let mut l2sum = 0.0;
            for k in 0..data[0].len(){
                l2sum += (data[i][k]-data[j][k]).powi(2);
            }
            dist_mat[[i,j]] = l2sum.sqrt();
            dist_mat[[j,i]] = dist_mat[[i,j]];
        }
    };
    dist_mat
}

fn read_labels(path:&str) -> Array1<i32>{
    let file = File::open(path).unwrap();
    let buffer = BufReader::new(file);
    let mut res = Vec::new();
    for line in buffer.lines() {
        let line = line.unwrap();
        res.push(line.parse::<i32>().unwrap());
    };
    Array1::from(res)
}


fn main() -> Result<(), Box<dyn Error>> {
	let data_path = env::args().nth(1).expect("no file name given");
    let label_path = env::args().nth(2).expect("no file name given");

    let dist_mat = read_data(&data_path);
    let labels = read_labels(&label_path);

    let mut medoids = (0 .. 10).collect::<Vec<usize>>();

	let start = Instant::now();
	let (loss, _, iter, swaps) : (f64, _, _, _)  =  labeledpam(&dist_mat, &labels, &mut medoids, 2, 100);
	let duration = start.elapsed();
	println!("FasterPAM final loss: {}", loss);
	println!("FasterPAM swaps performed: {}", swaps);
	println!("FasterPAM number of iterations: {}", iter);
	println!("FasterPAM optimization time: {:?}", duration);
	Ok(())
}