//! Adapter trait for accessing different types of arrays.
//!
//! Includes adapters for `ndarray::Array2` and a serialized lower triangular matrix in a `Vec`.

/// Adapter trait for accessing different types of arrays
#[allow(clippy::len_without_is_empty)]
pub trait ArrayAdapter<N: Copy> {
	/// Get the length of an array structure
	fn len(&self) -> usize;
	/// Verify that it is a square matrix
	fn is_square(&self) -> bool;
	/// Get the contents at cell x,y
	fn get(&self, x: usize, y: usize) -> N;
}

/// Adapter trait for using `ndarray::Array2`
#[cfg(feature = "ndarray")]
impl<N: Copy> ArrayAdapter<N> for ndarray::Array2<N> {
	#[inline]
	fn len(&self) -> usize {
		self.shape()[0]
	}
	#[inline]
	fn is_square(&self) -> bool {
		self.shape()[0] == self.shape()[1]
	}
	#[inline]
	fn get(&self, x: usize, y: usize) -> N {
		self[[x, y]]
	}
}

/// Lower triangular matrix in serial form (without diagonal)
///
/// ## Example
/// ```
/// let data = kmedoids::arrayadapter::LowerTriangle { n: 4, data: vec![1, 2, 3, 4, 5, 6] };
/// let mut meds = vec![0, 1];
/// let (loss, numswap, numiter, assignment) = kmedoids::fasterpam(&data, &mut meds, 10);
/// println!("Loss is {}", loss);
/// ```
#[derive(Debug, Clone)]
pub struct LowerTriangle<N> {
	/// Matrix size
	pub n: usize,
	// Matrix data, lower triangular form without diagonal
	pub data: Vec<N>,
}
/// Adapter implementation for LowerTriangle
impl<N: Copy + num_traits::Zero> ArrayAdapter<N> for LowerTriangle<N> {
	#[inline]
	fn len(&self) -> usize {
		self.n
	}
	#[inline]
	fn is_square(&self) -> bool {
		self.data.len() == (self.n * (self.n - 1)) >> 1
	}
	#[inline]
	fn get(&self, x: usize, y: usize) -> N {
		match x.cmp(&y) {
			std::cmp::Ordering::Less => self.data[((y * (y - 1)) >> 1) + x],
			std::cmp::Ordering::Greater => self.data[((x * (x - 1)) >> 1) + y],
			std::cmp::Ordering::Equal => N::zero(),
		}
	}
}
