use std::{fmt::Display, fmt::Debug, ops::AddAssign};

use num_traits::{AsPrimitive, One, Signed, Zero};

use crate::arrayadapter::LabelAdapter;


macro_rules! trait_combiner {
    ($combination_name: ident) => {
        pub trait $combination_name {}
        impl<T> $combination_name for T {}
    };
    ($combination_name: ident: $t: tt $(+ $ts: tt)*) => {
        pub trait $combination_name: $t $(+ $ts)* {}
        impl<T: $t $(+ $ts)*> $combination_name for T {}
    };
}
trait_combiner!(Label: Zero + One + Signed + Max + PartialOrd + Clone + Copy + IntoIndex + FromIndex + Debug + Display);

pub trait Max {
    const MAX: Self;
}
macro_rules! max {
    ($($type:ident),*) => {
        $(impl Max for $type {
            const MAX: Self = <$type>::MAX;
        })*
    };
}

max!(u16, u32, u64, i16, i32, i64);

pub trait IntoIndex:Zero + PartialOrd + Clone + Copy + One{
    /// # Panics
    /// This will panic if the resulting value won't fit into a `usize`, typically because it
    /// contains a fraction or is outside of the range supported by `usize`.
    
    fn into_index(self) -> usize;

    const MAX: Self;
}

macro_rules! into_index {
    ($($type:ident),*) => {
        $(impl IntoIndex for $type {
            fn into_index(self) -> usize {
                self as usize
            }
            const MAX: Self = <$type>::MAX;
        })*
    };
}

into_index!(u16, u32, u64, i16, i32, i64);

pub trait FromIndex {
    fn from_index(index: usize) -> Self;
}

impl<T: Copy+'static> FromIndex for T where usize: AsPrimitive<T>{
    fn from_index(index: usize) -> T {
        index.as_()
    }
}

// macro_rules! from_index {
//     ($($type:ident),*) => {
//         $(impl FromIndex for $type {
//             fn from_index(index: usize) -> $type {
//                 index as $type
//             }
//         })*
//     };
// }

// from_index!(u16, u32, u64, i16, i32, i64);

/// Information kept about the clustering
#[derive(Debug)]
pub(crate) struct CluRec<'a, N> 
where
	N: Label,
{
	pub(crate) meds: & 'a mut [usize],
	pub(crate) clus_labels: Vec<N>,
	pub(crate) label_counts: Vec<usize>,
	pub(crate) clusters_per_label: Vec<usize>,
	pub(crate) unlabeled_clusters: usize,
}

impl<'a, N:Label> CluRec<'a, N> 
{
	pub(crate) fn new(meds: &'a mut [usize], labels: &dyn LabelAdapter<N>, label_count:usize) -> CluRec<'a, N>
	{
		let meds_len = meds.len();
		let clus_labels:Vec<N> = meds.iter().map(|m|{
			labels.get(*m)
		}).collect();
		let mut clu_rec = CluRec {
			meds,
			clus_labels,
			label_counts: vec![usize::zero(); meds_len],
			clusters_per_label: vec![usize::zero(); label_count],
			unlabeled_clusters: 0,
		};
		clu_rec.update_cluster_per_label();
		return clu_rec;
	}

	pub(crate) fn len(&self) -> usize {
		self.meds.len()
	}

	pub(crate) fn no_labels(&self) -> usize {
		self.clusters_per_label.len()
	}

	// update all the label information for the clusters
	pub(crate) fn update_labels(&mut self){
		// reset labels if no labeled point is present
		for (i, l) in self.clus_labels.iter_mut().enumerate(){
			if self.label_counts[i] == 0{
				*l = -N::one();
			}
		}
		self.update_cluster_per_label()
	}

	// update the number of clusters per label
	pub(crate) fn update_cluster_per_label(&mut self){
		self.clusters_per_label.fill(0);
		self.unlabeled_clusters = 0;
		for label in self.clus_labels.iter(){
			if label >= &N::zero(){
				self.clusters_per_label[label.clone().into_index()] += 1;
			} else {
				self.unlabeled_clusters += 1;
			}
		}
	}
}

impl <'a, N> Display for CluRec<'a, N>
where
	N: Label,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "CluRec:\n medoids: {:?},\n clu_lab: {:?},\n lab_cnt: {:?},\n cl_p_la: {:?},\n unl_clu: {}", self.meds, self.clus_labels, self.label_counts, self.clusters_per_label, self.unlabeled_clusters)
	}
}

#[inline]
pub(crate) fn can_uncolor<C: Label>(cluster_records: &CluRec<C>, label:C) -> bool
{
	if cluster_records.len() == cluster_records.no_labels(){
		return false;
	}
	if label >= C::zero() && cluster_records.clusters_per_label[label.into_index()] > 1{
		return true;
	}
	return cluster_records.unlabeled_clusters > 0;
}

// find the labeled min (index, value and label)
#[inline]
pub(crate) fn find_label_min<'a, L, C>(j_label:C, cluster_records:&CluRec<C>, ploss:&mut Vec<L>, closs:&Vec<L>, acc:L, cacc:&Vec<L>) -> (usize, C, L)
	where
		L: PartialOrd + Clone + Zero + 'a + AddAssign + Copy,
		C: Label,
{
	let mut base_label = j_label;
	let mut alt_label = -C::one();
	let mut min = L::zero();
	let mut min2 = L::zero();
	let mut best = 0;
	let mut best_label:C = -C::one();
	let mut loss = L::zero();
	// if the candidate is not labeled, find the best two colors
	if base_label == -C::one() {
		for (i, a) in cacc.iter().enumerate() {
			if *a < min {
				min2 = min;
				alt_label = base_label;
				min = *a;
				base_label = C::from_index(i);
			} else if *a < min2 {
				min2 = *a;
				alt_label = C::from_index(i);
			}
		}
	}
	for (i, i_loss) in ploss.iter().enumerate() {
		let cluster_label = cluster_records.clus_labels[i];
		if j_label >= C::zero() {
			if j_label != cluster_label && !can_uncolor(cluster_records, cluster_label){
				// labels incompatible
				continue;
			} else if base_label == cluster_label {
				debug_assert!(j_label == cluster_label, "Incompatible labels");
				let temp_loss = *i_loss + closs[i] + acc + cacc[cluster_label.into_index()];
				if temp_loss < loss {
					loss = temp_loss;
					best = i;
					best_label = base_label;
				}
			} else {
				let temp_loss = *i_loss + acc + cacc[j_label.into_index()];
				if temp_loss < loss {
					loss = temp_loss;
					best = i;
					best_label = base_label;
				}
			}
		} else if can_uncolor(cluster_records, cluster_label) {
			let mut local_best = base_label;
			if local_best != cluster_label && closs[i] < L::zero(){
				local_best = if min < cacc[cluster_label.into_index()] + closs[i] {local_best} else {cluster_label};
			}else if local_best == cluster_label && closs[i] > L::zero(){
				local_best = if cacc[local_best.into_index()] + closs[i] < min2 {local_best} else {alt_label};
			}
			let mut temp_loss = *i_loss + acc;
			if local_best == cluster_label {
				temp_loss += closs[i];
			}
			// add the benefit of the best label
			if local_best >= C::zero() {
				temp_loss += cacc[local_best.into_index()];
			}
			if temp_loss < loss {
				loss = temp_loss;
				best = i;
				best_label = local_best;
			}
		} else {
			// cluster can not be relabeled
			// cluster can not be unlabeled
			debug_assert!(cluster_label >= C::zero(), "Cluster label is negative");
			let temp_loss = *i_loss + closs[i] + acc + cacc[cluster_label.into_index()];
			if temp_loss < loss {
				loss = temp_loss;
				best = i;
				best_label = cluster_label;
			}
		}
	}
	return (best, best_label, loss);
}