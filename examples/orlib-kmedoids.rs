// If you intend to use this for benchmarking, add a shuffle step and use fixed random seeds.
// Because of ties in ORLib data sets, order matters even when you use fixed random seeds.
use kmedoids::{fasterpam, random_initialization};
use ndarray::Array2;
use num_traits::{NumOps, Zero};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::num::ParseIntError;
use std::time::Instant;

// problem to solve
struct Problem<N> {
	data: Array2<N>,
	k: usize,
}

fn all_pairs_shortest_path<N>(mat: &mut Array2<N>)
where
	N: NumOps + PartialOrd + Copy,
{
	let n = mat.shape()[0];
	assert_eq!(n, mat.shape()[1]);
	// Floyd's all pairs shortest path
	for i in 0..n {
		for x in 0..n {
			for y in (x + 1)..n {
				let m = mat[[x, i]] + mat[[i, y]];
				if m < mat[[x, y]] {
					mat[[y, x]] = m;
					mat[[x, y]] = m;
				}
			}
		}
	}
}

fn read_orlib<R: Read, T>(io: R, default_value: T) -> Result<Problem<T>, Box<dyn Error>>
where
	T: NumOps + Zero + PartialOrd + Copy + std::str::FromStr,
	// Rust black magic:
	T::Err: std::error::Error + 'static,
{
	println!("Reading PMED file");
	let mut buffer = BufReader::new(io);

	// First line is n, p, k. We ignore n.
	let mut first_line = String::new();
	buffer
		.read_line(&mut first_line)
		.expect("Unable to read data file");

	let first = first_line
		.split_whitespace()
		.map(|x| x.parse::<usize>())
		.collect::<Result<Vec<usize>, ParseIntError>>()?;
	assert_eq!(first.len(), 3);
	let n: usize = first[0];
	let k: usize = first[2];

	let mut mat = Array2::<T>::from_elem((n, n), default_value);
	for line in buffer.lines() {
		let ln = line?;
		let mut split = ln.split_whitespace();
		let x1: usize = split.next().expect("no x").parse()?;
		let x2: usize = split.next().expect("no y").parse()?;
		assert_ne!(x1, x2);
		let d: T = split.next().expect("no v").parse()?;
		mat[[x1 - 1, x2 - 1]] = d;
		mat[[x2 - 1, x1 - 1]] = d;
	}
	for x in 0..n {
		mat[[x, x]] = T::zero();
	}
	let start = Instant::now();
	all_pairs_shortest_path(&mut mat);
	let duration = start.elapsed();
	println!("All-pairs shortest path took: {:?}", duration);
	Ok(Problem { data: mat, k })
}

fn main() -> Result<(), Box<dyn Error>>{
	let nam = env::args().nth(1).expect("no file name given");
	let prob = read_orlib(File::open(nam)?, i32::MAX)?;
	let mut rand = rand::thread_rng();
	let start = Instant::now();
	let mut meds = random_initialization(prob.data.shape()[0], prob.k, &mut rand);
	let (loss, _, iter, swaps) : (i64, _, _, _) = fasterpam(&prob.data, &mut meds, 100);
	let duration = start.elapsed();
	println!("FasterPAM final loss: {}", loss);
	println!("FasterPAM swaps performed: {}", swaps);
	println!("FasterPAM number of iterations: {}", iter);
	println!("FasterPAM optimization time: {:?}", duration);
	Ok(())
}
