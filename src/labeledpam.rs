use crate::arrayadapter::ArrayAdapter;
use crate::arrayadapter::LabelAdapter;
use crate::util::*;
use crate::labelutils::*;
use core::ops::AddAssign;
use std::fmt::Display;
use num_traits::{Signed, Zero};
use std::convert::From;

/// Run the LabeledPAM algorithm.
///
/// If used multiple times, it is better to additionally shuffle the input data,
/// to increase randomness of the solutions found and hence increase the chance
/// of finding a better solution.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the number of iterations needed
/// * the number of swaps performed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn labeledpam<M, N, L,C, O>(
	mat: &M,
	labels: &O,
	med: &mut [usize],
	no_labels: usize,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N> + Copy + Display,
	M: ArrayAdapter<N>,
	C: Label,
	O: LabelAdapter<C>,
{
	let (n, k) = (mat.len(), med.len());
	assert!(mat.len() == labels.len(), "Labels and matrix must have the same length");
	assert!(no_labels <= k, "Too many labels {} for the number of medoids {}", no_labels, k);
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let mut cluster_records = CluRec::<C>::new(med, labels, no_labels);
	let (mut loss, mut data): (L, Vec<Rec<N>>) = initial_assignment(mat, labels, &mut cluster_records);
	debug_assert_assignment(mat, &cluster_records.meds, &data);
	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&mut data, &mut removal_loss, mat, labels, &mut cluster_records);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
	while iter < maxiter {
		iter += 1;
		let (swaps_before, lastloss) = (n_swaps, loss.clone());
		for j in 0..n {
			if j == lastswap {
				break;
			}
			if j == cluster_records.meds[data[j].near.i as usize] {
				continue; // This already is a medoid
			}
			let (b, l, change) = find_best_swap(mat, labels, &removal_loss, &data, &cluster_records, j);
			if change >= L::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			loss = do_swap(mat, labels, &mut data, &mut cluster_records, j, b, l);
			update_removal_loss(&mut data, &mut removal_loss, mat, labels, &mut cluster_records);
		}
		if n_swaps == swaps_before || loss >= lastloss {
			break; // converged
		}
	}
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}


/// Perform the initial assignment to medoids
#[inline]
pub(crate) fn initial_assignment<M, N, L, C, O>(mat: &M, labels: &O, cluster_records: &mut CluRec<C>) -> (L, Vec<Rec<N>>)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Zero + PartialOrd + Clone + From<N>,
	M: ArrayAdapter<N>,
	C: Label,
	O: LabelAdapter<C>,
{
	let (n, k) = (mat.len(), cluster_records.len());
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid K");
	assert!(k <= n, "k must be at most N");
	let mut data = vec![Rec::<N>::empty(); mat.len()];

	let loss = data
		.iter_mut()
		.enumerate()
		.map(|(i, cur)| {
			*cur = Rec::new(u32::MAX, N::zero(), u32::MAX, N::zero());
			for (m, &me) in cluster_records.meds.iter().enumerate() {
				if !is_valid_sec_pair(i, labels, m, cluster_records){
					continue;
				}
				let d = mat.get(i, me);
				if i == me || ((d < cur.near.d || cur.near.i == u32::MAX) && is_valid_pair(i, labels, m, cluster_records)) {
					cur.seco = cur.near.clone();
					cur.near = DistancePair { i: m as u32, d };
				} else if cur.seco.i == u32::MAX || d < cur.seco.d {
					cur.seco = DistancePair { i: m as u32, d };
				}
			}
			debug_assert!(cur.near.i != u32::MAX, "No medoid found for object {}", i);
			if labels.get(i)>= C::zero(){
				cluster_records.label_counts[cur.near.i as usize] += 1;
			}
			L::from(cur.near.d.clone())
		})
		.reduce(L::add)
		.unwrap();
	cluster_records.update_cluster_per_label();
	(loss, data)
}

/// Find the best swap for object j - FastPAM version
#[inline]
pub(crate) fn find_best_swap<M, N, L, C, O>(
	mat: &M,
	labels: &O,
	removal_loss: &[L],
	data: &[Rec<N>],
	cluster_records: &CluRec<C>,
	j: usize,
) -> (usize, C, L)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N> + Copy,
	M: ArrayAdapter<N>,
	C: Label,
	O: LabelAdapter<C>,
{
	let j_label = labels.get(j);
	let mut ploss = removal_loss.to_vec();
	let mut closs = vec![L::zero(); cluster_records.len()];
	let mut acc = L::zero();
	let mut cacc = vec![L::zero(); cluster_records.no_labels()];
	for (o, reco) in data.iter().enumerate() {
        let o_label = labels.get(o);

		// Skip object and medoid are not label compatible
		if !(j_label == o_label || j_label < C::zero() || o_label < C::zero()){
			continue;
		}
		let sec_valid = reco.seco.i != u32::MAX;
		let doj = mat.get(o, j);
		// there is an alternative medoid for o
		if sec_valid {
			if doj < reco.near.d {
				if o_label >= C::zero() {
					cacc[o_label.into_index()] += L::from(doj) - L::from(reco.near.d.clone());
					closs[reco.near.i as usize] += L::from(reco.near.d.clone()) - L::from(reco.seco.d.clone());
				} else {
					acc += L::from(doj) - L::from(reco.near.d.clone());
					ploss[reco.near.i as usize] += L::from(reco.near.d.clone()) - L::from(reco.seco.d.clone());
				}
			} else if doj < reco.seco.d {
				// object with label
				if o_label >= C::zero() {
					closs[reco.near.i as usize] += L::from(doj) - L::from(reco.seco.d.clone());
				} else {
					ploss[reco.near.i as usize] += L::from(doj) - L::from(reco.seco.d.clone());
				}
			} 
		// no alternative medoid for o
		} else {
			let benefit = L::from(doj.clone()) - L::from(reco.near.d.clone());

			if benefit < L::zero() {
				ploss[reco.near.i as usize] += benefit;
				cacc[o_label.into_index()] += benefit;
			} else {
				ploss[reco.near.i as usize] += L::from(doj.clone());
			}
		}
	}
	return find_label_min(j_label, cluster_records, &mut ploss, &closs, acc, &cacc);
}

/// Update the loss when removing each medoid
pub(crate) fn update_removal_loss<N, L, M, C, O>(data: &mut [Rec<N>], loss: &mut [L], mat: &M, labels: &O, cluster_records: &mut CluRec<C>)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Clone + Zero + From<N>,
	C: Label,
	O: LabelAdapter<C>,
	M: ArrayAdapter<N>,
{
	loss.fill(L::zero()); // stable since 1.50
	for (i, rec) in data.iter_mut().enumerate() {
		if rec.seco.i == u32::MAX {
			if can_uncolor(cluster_records, labels.get(i)){
				rec.seco = update_second_nearest(mat, cluster_records, labels, rec.near.i as usize, u32::MAX as usize, i, N::zero());
			}
		} else if !is_valid_sec_pair(i, labels, rec.seco.i as usize, cluster_records){
			rec.seco = update_second_nearest(mat, cluster_records, labels, rec.near.i as usize, u32::MAX as usize, i, N::zero());
		}
		loss[rec.near.i as usize] += L::from(rec.seco.d.clone()) - L::from(rec.near.d.clone());
		// as N might be unsigned
	}
}

/// Update the second nearest medoid information
/// Called after each swap.
#[inline]
pub(crate) fn update_second_nearest<M, N, C, O>(
	mat: &M,
	cluster_records: &CluRec<C>,
	labels: &O,
	n: usize,
	b: usize,
	o: usize,
	doj: N,
) -> DistancePair<N>
where
	N: Zero + PartialOrd + Clone,
	M: ArrayAdapter<N>,
	C: Label,
	O: LabelAdapter<C>,
{
	// alternatively make sure that doj is 0 when b invalid?
	let mut s = if b == u32::MAX as usize{
									DistancePair::empty()
								} else {
									DistancePair::new(b as u32, doj.clone())
								};
	for (i, &mi) in cluster_records.meds.iter().enumerate() {
		if i == n || i == b || !is_valid_sec_pair(o, labels, i, cluster_records){
			continue;
		}
		let d = mat.get(o, mi);
		if d < s.d {
			s = DistancePair::new(i as u32, d);
		}
	}
	s
}

/// Perform a single swap
#[inline]
pub(crate) fn do_swap<M, N, L, C, O>(
	mat: &M,
	labels: &O,
	data: &mut [Rec<N>], 
	cluster_records: &mut CluRec<C>,
	j: usize,
	b: usize, 
	l: C,
) -> L
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N>,
	M: ArrayAdapter<N>,
	C: Label,
	O: LabelAdapter<C>,
{
	let n = mat.len();
	assert!(b < cluster_records.meds.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
	cluster_records.meds[b] = j;
	cluster_records.clus_labels[b] = l;
	data.iter_mut()
		.enumerate()
		.map(|(o, reco)| {
			// new medoid
			if o == j {
				if reco.near.i != b as u32 {
					if labels.get(j) >= C::zero() {
						cluster_records.label_counts[reco.near.i as usize] -= 1;
						cluster_records.label_counts[b] += 1;
					}
					reco.seco = reco.near.clone();
				}
				reco.near = DistancePair::new(b as u32, N::zero());
				return L::zero();
			}
			let doj = mat.get(o, j);
			let obj_color = labels.get(o);
			let valid_pair = is_valid_pair(o, labels, b, cluster_records);
			// Nearest medoid is gone:
			if reco.near.i == b as u32 {
				if doj < reco.seco.d || reco.seco.i == u32::MAX{
					reco.near = DistancePair::new(b as u32, doj);
				} else {
					if obj_color >= C::zero() {
						cluster_records.label_counts[reco.near.i as usize] -= 1;
						cluster_records.label_counts[b] += 1;
					}
					reco.near = reco.seco.clone();
					reco.seco = update_second_nearest(mat, &cluster_records, labels, reco.near.i as usize, if valid_pair{b} else {u32::MAX as usize}, o, if valid_pair {doj} else {N::zero()});
				}
			} else {
				// nearest not removed
				if doj < reco.near.d && valid_pair{
					if obj_color >= C::zero() {
						cluster_records.label_counts[reco.near.i as usize] -= 1;
						cluster_records.label_counts[b] += 1;
					}
					if reco.near.d < reco.seco.d || reco.seco.i == u32::MAX{
						reco.seco = reco.near.clone();
					}
					reco.near = DistancePair::new(b as u32, doj);
				} else if doj < reco.seco.d && valid_pair{
					reco.seco = DistancePair::new(b as u32, doj);
				} else if reco.seco.i == b as u32 {
					// second nearest was replaced
					reco.seco = update_second_nearest(mat, cluster_records, labels,  reco.near.i as usize, if valid_pair{b} else {u32::MAX as usize}, o, if valid_pair {doj} else {N::zero()});
				}
			}
			L::from(reco.near.d.clone())
		})
		.reduce(L::add)
		.unwrap()
}

#[inline]
pub(crate)  fn is_valid_pair<O, C>(obj:usize, labels:&O, med:usize, cluster_records: &CluRec<C>) -> bool
where 
	C: Label,
	O: LabelAdapter<C>,
{
	if labels.get(obj) < C::zero() || labels.get(obj) == cluster_records.clus_labels[med]{
		return true
	}
	return false
}

#[inline]
pub(crate) fn is_valid_sec_pair<O, C>(obj:usize, labels:&O, med:usize, cluster_records: &CluRec<C>) -> bool
where 
	C: Label,
	O: LabelAdapter<C>,
{
	if cluster_records.label_counts[med] == 0{
		return true
	} 
	if labels.get(obj) < C::zero() || labels.get(obj) == cluster_records.clus_labels[med]{
		return true
	}
	return false
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, labeledpam, silhouette, util::assert_array, arrayadapter::LabelList};

	#[test]
	fn testlabeledpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		};
		let labels = LabelList{
			data: vec![0, 1, -1, 0,1],
		};
		let mut meds = vec![0, 1, 2];
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = labeledpam(&data, &labels, &mut meds, 2, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 2, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testlabeledpam_complex() {
		let data = LowerTriangle {
			n: 20,
			data: vec![1.1879279423816165, 1.2947944788394952, 1.1222122549352096, 1.414307734436693, 1.117735446934769, 1.4388925189516457, 1.8327068332077852, 1.2158016984120257, 1.3989851936005935, 1.426547267449687, 1.423286850371007, 1.230692652011346, 1.1790880402125141, 1.331246985115263, 1.242469101916525, 1.5233457287445324, 1.2908669921560452, 1.8876257455458059, 1.2976864095676097, 1.28487361618566, 1.4508818160764434, 1.5802171015384037, 0.8246468423211168, 1.0770759536026042, 1.1708931778372325, 0.9322451613022035, 1.1341982534581407, 1.392362374901237, 1.3014441889924953, 1.380847866822644, 1.4837217164280534, 1.0976835385015102, 1.3737324456385667, 1.4188274838941966, 1.574619874391499, 1.3076407523174982, 1.4962581844810656, 1.0476639063206692, 1.4238735613602689, 1.1635477475189677, 0.7694415463517579, 1.3918285774375523, 1.063128604792607, 1.2333934305393666, 1.2550881177869355, 1.573817074139927, 1.2663276887473867, 1.5861386067390943, 1.0360849241277867, 1.120456870500434, 1.5190479059695674, 0.8414283805459416, 1.0751032358608699, 1.330087992516393, 1.066536888222489, 1.3650366481566873, 1.003694845959014, 0.7828022614145911, 1.0832759356984953, 1.171191891026838, 1.1282537012411014, 1.7277865316441579, 0.8549108047329211, 1.1457272037679442, 1.240663183247637, 1.4423895400260947, 1.3893649737130662, 1.2598886243958187, 1.1768299594812994, 1.328280986475995, 1.0354059760712688, 1.3516242570674242, 1.568616993928676, 1.0854429766124625, 0.7750454774497934, 1.0948732530122414, 1.252213898560902, 0.9468836176314643, 1.3448066508962166, 1.2772220139970132, 1.1479971281377912, 1.3041195803487524, 0.9882425622445932, 1.3001232706562216, 1.4785655148448364, 1.171230705441449, 0.8671434330685394, 0.9786345009553221, 1.1569454993678685, 1.0387607832316152, 0.3205960525928075, 1.2107621032076297, 1.0385514710713455, 1.2673254235210827, 1.2379698140034194, 1.2429450450953818, 1.5083996115042602, 1.1312208560481458, 1.269908366750285, 1.4467857073756032, 0.8137235116395952, 1.1650543924541379, 1.1866214363817738, 1.2181547292740962, 1.1767000900330242, 1.232500327154175, 1.1959178366794556, 1.266381726261405, 1.2633612352409676, 1.4822434155611741, 1.4590813927580275, 1.7768483317050352, 1.2121259410275416, 1.1047101645940685, 1.432259181864977, 1.6052975739586235, 0.6735500379651631, 1.1173418534512762, 1.259042225384029, 1.2878098394261042, 1.6456323858999995, 1.2769431147066836, 1.2498535260764427, 1.3712828605189429, 0.5611834016961179, 1.2879097421171428, 1.3456173277784909, 0.9089703645095416, 1.2401482903915062, 0.9608180923773559, 0.9933544093135666, 0.9862788845601757, 0.7881557671993221, 0.7527908651132129, 1.1663604629500948, 1.2251300803998495, 1.0551649762366742, 1.1131303105702126, 1.186712419672403, 1.492431874233841, 1.4707166931388924, 1.0766536679320913, 1.2847470336894629, 1.4676235769681403, 1.745418679067234, 1.236396697454031, 1.4434625116329443, 1.4390754491687048, 1.5903499698717873, 1.4182136490771975, 1.1658651730360627, 1.648225792104958, 1.4603645957781444, 1.2552454721093, 0.8438116073585317, 0.9760624968447917, 1.1268639782740255, 1.1003871390369602, 1.0395118723802517, 1.535861802748197, 0.9907646849743126, 1.1470505643206161, 1.0143082132176893, 1.4796203031220336, 0.5369594436234569, 0.9025569854817785, 0.9750677527909336, 0.9990933027634951, 0.8119252089883857, 1.0475345658728665, 1.283886669174992, 1.279498987285217, 1.302419837117496, 1.470415666981714, 1.4438291975904356, 1.5097560390227858, 1.534509918279619, 1.5198757168668877, 1.4839770406541033, 1.111839891171417, 1.2384065273386382, 1.4715621877693774, 1.418948754613172, 0.8383703349546715, 0.8439716659624814, 1.20542846925377, 1.5477559427822039, 1.4191366814755169, 1.5608452845841128, 1.1293346383788976],
		};
		let labels = LabelList{
			data: vec![0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (f64, _, _, _) = labeledpam(&data, &labels, &mut meds, 2, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_array(assi, vec![0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0], "assignment not as expected");
		assert_eq!(loss, 0.0, "loss not as expected");
		assert_eq!(n_swap, 2, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testlabeledpam_single_cluster() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		};
		let labels = LabelList{
			data: vec![0, 0, -1,-1,-1],
		};
		let mut meds = vec![1]; // So we need one swap
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = labeledpam(&data, &labels, &mut meds, 1, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 14, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 1, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 0, 0], "assignment not as expected");
		assert_array(meds, vec![0], "medoids not as expected");
		assert_eq!(sil, 0., "Silhouette not as expected");
	}
}
