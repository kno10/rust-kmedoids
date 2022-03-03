use kmedoids::{fasterpam, random_initialization};
use ndarray::{Array2, Ix};
use std::env;
use std::error::Error;
use std::time::Instant;
use std::fs::File;
use std::io::Read;
use byteorder::{BigEndian, ReadBytesExt};

// problem to solve
struct Problem<N> {
	data: Array2<N>,
}

fn read_mnist(path: String) -> Result<Problem<f64>, Box<dyn Error>>
{
	println!("Reading MNIST file");
	// Read whole file in memory
	let mut content: Vec<u8> = Vec::new();
	let mut file = {
		let mut fh = File::open(&path)
			.unwrap_or_else(|_| panic!("Unable to find path to images at {:?}.", &path));
		let _ = fh
			.read_to_end(&mut content)
			.unwrap_or_else(|_| panic!("Unable to read whole file in memory ({})", &path));
		// The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
		// used with a `Vec` directly, it requires a slice.
		&content[..]
	};
	file.read_u32::<BigEndian>().unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
	let len:usize = file.read_u32::<BigEndian>().unwrap_or_else(|_| panic!("Unable to length from {:?}.", path)) as usize;
	let rows:usize = file.read_u32::<BigEndian>().unwrap_or_else(|_| panic!("Unable to number of rows from {:?}.", path)) as usize;
	let cols:usize = file.read_u32::<BigEndian>().unwrap_or_else(|_| panic!("Unable to number of columns from {:?}.", path)) as usize;
	let img_vec = file.to_vec();
	let train_data = Array2::from_shape_vec((len, rows*cols), img_vec)
		.expect("Error converting images to Array3 struct")
		.map(|x| *x as f64 / 256.0);
	let mut mat = Array2::<f64>::from_elem((len, len), 0.0);
	let start = Instant::now();
	for m in 0..len as Ix {
		for n in m..len as Ix {
			let mut l2sum = 0.0;
			let dist = &train_data.row(m) - &train_data.row(n);
			l2sum += (dist).dot(&dist).sqrt();
			mat[[m, n]] = l2sum;
			mat[[n, m]] = l2sum;
		}
	}
	let duration = start.elapsed();
	println!("distance matrix for mnist: {:?}", duration);
	Ok(Problem { data: mat })
}

fn main() -> Result<(), Box<dyn Error>> {
	let path = env::args().nth(1).expect("no file name given");
	let prob = read_mnist(path).unwrap();
	let mut rand = rand::thread_rng();
	let start = Instant::now();
	let mut meds = random_initialization(prob.data.shape()[0], 10, &mut rand);
	let (loss, _, iter, swaps) : (f64, _, _, _)  = fasterpam(&prob.data, &mut meds, 100);
	let duration = start.elapsed();
	println!("FasterPAM final loss: {}", loss);
	println!("FasterPAM swaps performed: {}", swaps);
	println!("FasterPAM number of iterations: {}", iter);
	println!("FasterPAM optimization time: {:?}", duration);
	Ok(())
}