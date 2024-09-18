//! k-Medoids Clustering with the FasterPAM Algorithm
//!
//! For details on the implemented FasterPAM algorithm, please see:
//!
//! Erich Schubert, Peter J. Rousseeuw  
//! **Fast and Eager k-Medoids Clustering:  
//! O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**  
//! Information Systems (101), 2021, 101804  
//! <https://doi.org/10.1016/j.is.2021.101804> (open access)
//!
//! Erich Schubert, Peter J. Rousseeuw:  
//! **Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms**  
//! In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.  
//! <https://doi.org/10.1007/978-3-030-32047-8_16>  
//! Preprint: <https://arxiv.org/abs/1810.05691>
//!
//! This is a port of the original Java code from [ELKI](https://elki-project.github.io/) to Rust.
//! But it does not include all functionality in the original benchmarks.
//!
//! If you use this in scientific work, please consider citing above articles.
//!
//! ## Example
//!
//! Given a dissimilarity matrix of size 4 x 4, use:
//! ```
//! let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
//! let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
//! let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&data, &mut meds, 100);
//! println!("Loss is: {}", loss);
//! ```
mod initialization;
pub mod arrayadapter;
mod silhouette;
mod alternating;
mod util;
#[cfg(feature = "FasterPAM")]
mod fasterpam;
#[cfg(feature = "FasterMSC")]
mod fastermsc;
#[cfg(feature = "DynMSC")]
mod dynmsc;
#[cfg(feature = "parallel")]
mod par_fasterpam;
#[cfg(feature = "parallel")]
mod par_silhouette;

#[cfg(feature = "FastPAM1")]
mod fastpam1;
#[cfg(feature = "FastMSC")]
mod fastmsc;
#[cfg(feature = "PAM")]
mod pam;
#[cfg(feature = "PAMSIL")]
mod pamsil;
#[cfg(feature = "PAMMEDSIL")]
mod pammedsil;

pub use crate::initialization::*;
pub use crate::arrayadapter::ArrayAdapter;
pub use crate::silhouette::*;
pub use crate::alternating::*;

#[cfg(feature = "FasterPAM")]
pub use crate::fasterpam::*;
#[cfg(feature = "FasterMSC")]
pub use crate::fastermsc::*;
#[cfg(feature = "DynMSC")]
pub use crate::dynmsc::*;
#[cfg(feature = "parallel")]
pub use crate::par_fasterpam::*;
#[cfg(feature = "parallel")]
pub use crate::par_silhouette::*;

#[cfg(feature = "FastPAM1")]
pub use crate::fastpam1::*;
#[cfg(feature = "FastMSC")]
pub use crate::fastmsc::*;
#[cfg(feature = "PAM")]
pub use crate::pam::*;
#[cfg(feature = "PAMSIL")]
pub use crate::pamsil::*;
#[cfg(feature = "PAMMEDSIL")]
pub use crate::pammedsil::*;
