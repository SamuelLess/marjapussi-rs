//! Marjapussi rules engine.

pub mod bits;
pub mod game;
#[cfg(not(feature = "public-api"))]
mod inline;
mod storage;
#[cfg(not(feature = "public-api"))]
pub use inline::{Filler, InlineVec};
