# Changelog

## kmedoids 0.5.2 (2024-09-10)

- use Clone instead of Copy to better support arbitrary precision
- resolve some clippy warnings
- bump ndarray version (no changes)

## kmedoids 0.5.1 (2024-03-14)

- DynMSC: best loss reported incorrectly if best k=2
- add minimum k parameter
- bump rayon version (no changes)

## kmedoids 0.5.0 (2023-12-10)

- add DynMSC with automatic cluster number selection
- move the check for numerical instability out of the loop
  in all the "faster" variants, as we do no longer do best-first
- bump rayon to 1.8, no changes
- bump byteorder to 1.5, no changes, in example only

## kmedoids 0.4.3 (2023-04-20)

- fix bug in silhouette evaluation for k > 2

## kmedoids 0.4.2 (2023-03-07)

- bumped rayon to 1.7, with no changes
- add CITATION.cff

## kmedoids 0.4.1 (2022-09-24)

- drop a leftover println, remove Display/Debug traits
- optimize marginally the MSC loss function computation

## kmedoids 0.4.0 (2022-09-24)

- add clustering by optimizing the Silhouette: PAMSIL
- add medoid silhouette
- add medoid silhouette clustering: PAMMEDSIL, FastMSC, FasterMSC

## kmedoids 0.3.3 (2022-07-06)

- another bug fix in PAM BUILD, which ignored the first object

## kmedoids 0.3.2 (2022-06-25)

- bug fix in swap where the first k points were ignored instead of the centers (Lars Lenssen)
- small bug fix in PAM BUILD (noticable for tiny data sets with large k only)
- return less than k centers in BUILD if the total deviation already is 0 (less than k unique points)
- documentation improvement and packaging improvements in Python bindings

## kmedoids 0.3.1 (2022-04-05)

- no changes in Rust, only in the Python API

## kmedoids 0.3.0 (2022-03-27)

- reorder branches slightly (Lars Lenssen)
- no major changes in Rust, only in the Python API

## kmedoids 0.2.0 (2022-01-13)

- API change: allow input and loss types to differ
  (e.g., u32 input, i64 loss -- loss must be signed)
- removed safe_add trait -- use a higher precision loss type instead
- requires specifying the loss data type
- code modularization and cleanups
- fix: do not fail for k=1, but return the expected result
- add: added Silhouette index for evaluation
- add: rand_fasterpam with shuffled processing order
- add: par_fasterpam with parallelization using rayon (Lars Lenssen)

## kmedoids 0.1.6 (2021-09-02)

- update reference with published journal version
- update dependency versions
- improve some issue with ArrayBase

## kmedoids 0.1.5 (2021-01-31)

- add Alternating algorithm for reference
- move bench.rs to benches/

## kmedoids 0.1.4 (2021-01-31)

- remove forgotten println in pam reference method

## kmedoids 0.1.3 (2021-01-31)

- improve ndarray adapter to accept views
- relax version dependencies slightly

## kmedoids 0.1.2 (2020-12-24)

- reordered the returned values by importance to end users
- add FastPAM1 (slower, but exact same result as PAM)
- add original PAM (much slower, for reference)
- PAM BUILD and SWAP independently

## kmedoids 0.1.1 (2020-12-23)

- allow use with different array types (ndarray, lower triangular in a vec)

## kmedoids 0.1.0 (2020-12-22)

- initial port to Rust of the core FasterPAM functionality
- first Rust program ever, hence there likely is room for improvement
