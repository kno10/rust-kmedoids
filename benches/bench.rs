#![feature(test)]
//! Note that benchmarks can easily be misleading.
//! The Alternating algorithm comes out fastest - but it finds much worse solutions.
extern crate test;

use kmedoids::*;
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use test::{black_box, Bencher};

const SIZE : usize = 100;

#[bench]
fn bench_fasterpam(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _): (i32, _, _, _) = fasterpam(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_rand_fasterpam(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _): (i32, _, _, _) = rand_fasterpam(&mat, &mut med, 100, &mut rng);
		black_box(loss);
		black_box(assignment);
	});
}

#[cfg(feature = "parallel")]
#[bench]
fn bench_par_fasterpam(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _): (i32, _, _, _) = par_fasterpam(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_fastpam1(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _): (i32, _, _, _) = fastpam1(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_pam_swap(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _): (i32, _, _, _) = pam_swap(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_pam_build(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let (loss, assignment, _): (i32, _, _) = pam_build(&mat, 5);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_pam(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let (loss, assignment, _, _, _): (i32, _, _, _, _) = pam(&mat, 5, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_alternating(b: &mut Bencher) {
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((SIZE, SIZE), 0);
	for i in 0..SIZE {
		for j in (i + 1)..SIZE {
			let v = rng.gen_range(1..10000);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _): (i32, _, _) = alternating(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}
