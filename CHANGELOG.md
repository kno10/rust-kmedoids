# Changelog

## kmedoids 0.2.0 (2021-12-31)

- API change: allow input and loss types to differ
  (e.g., u32 input, i64 loss -- loss must be signed)
- removed safe_add trait -- use a higher precision loss type instead
- requires specifying the loss data type
- code modularization and cleanups
- fix: do not fail for k=1, but return the expected result
- add: added Silhouette index for evaluation
- add: rand_fasterpam with shuffled processing order

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
