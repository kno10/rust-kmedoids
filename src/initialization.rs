/// Random initialization (requires the `rand` crate)
///
/// This is simply a call to `rand::seq::index::sample`.
///
/// * `n` - size of the data set
/// * `k` - number of clusters to find
/// * `rng` - random number generator
///
/// returns a vector of medoid indexes in 0..n-1
///
/// ## Example
///
/// Given a dissimilarity matrix of size n x n, use:
/// ```
/// let mut meds = kmedoids::random_initialization(10, 2, &mut rand::thread_rng());
/// println!("Chosen medoids: {:?}", meds);
/// ```
#[cfg(feature = "rand")]
#[inline]
pub fn random_initialization(n: usize, k: usize, rng: &mut impl rand::Rng) -> Vec<usize> {
	rand::seq::index::sample(rng, n, k).into_vec()
}

/// Use the first objects as initial medoids.
///
/// * `k` - number of clusters to find
///
/// returns 0..k-1 as initial medoids
///
/// ## Example
///
/// Given a dissimilarity matrix of size n x n, use:
/// ```
/// let mut meds = kmedoids::first_k(2);
/// println!("Chosen medoids: {:?}", meds);
/// ```
#[inline]
pub fn first_k(k: usize) -> Vec<usize> {
	(0..k).collect()
}
