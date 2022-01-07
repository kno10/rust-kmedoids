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
