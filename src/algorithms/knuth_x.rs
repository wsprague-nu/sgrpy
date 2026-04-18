//! Knuth's Algorithm X and variants.
//!
//! For more information see [Wikipedia][1].
//!
//! [1]: https://en.wikipedia.org/wiki/Knuth's_Algorithm_X

mod algos;
mod links;
mod search;

pub use algos::{get_all, get_top, get_topn, get_topn_sc, get_topp};
