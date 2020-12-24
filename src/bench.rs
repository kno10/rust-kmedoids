//! Currently not used, because benchmarking is nightly only.
extern crate test;

use super::*;
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use test::{black_box, Bencher};

#[bench]
fn bench_fasterpam(b: &mut Bencher) {
	let n = 100;
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((n, n), 0);
	for i in 0..n {
		for j in (i + 1)..n {
			let v = rng.gen_range(1..100);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _) = fasterpam(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_fastpam1(b: &mut Bencher) {
	let n = 100;
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((n, n), 0);
	for i in 0..n {
		for j in (i + 1)..n {
			let v = rng.gen_range(1..100);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _) = fastpam1(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_pam_swap(b: &mut Bencher) {
	let n = 100;
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((n, n), 0);
	for i in 0..n {
		for j in (i + 1)..n {
			let v = rng.gen_range(1..100);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let mut med = vec![0, 1, 2, 3, 4];
		let (loss, assignment, _, _) = pam_swap(&mat, &mut med, 100);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_pam_build(b: &mut Bencher) {
	let n = 100;
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((n, n), 0);
	for i in 0..n {
		for j in (i + 1)..n {
			let v = rng.gen_range(1..100);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let (loss, assignment, _) = pam_build(&mat, 5);
		black_box(loss);
		black_box(assignment);
	});
}

#[bench]
fn bench_pam(b: &mut Bencher) {
	let n = 100;
	let mut rng = StdRng::seed_from_u64(42);
	let mut mat = Array2::<i32>::from_elem((n, n), 0);
	for i in 0..n {
		for j in (i + 1)..n {
			let v = rng.gen_range(1..100);
			mat[[i, j]] = v;
			mat[[j, i]] = v;
		}
	}
	b.iter(|| {
		let (loss, assignment, _, _, _) = pam(&mat, 5, 100);
		black_box(loss);
		black_box(assignment);
	});
}
