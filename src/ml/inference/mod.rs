pub mod engine;
pub mod rules;
pub mod terms;

pub use engine::{apply_hidden_set_constraints, hidden_set_constraints_enabled};
#[cfg(test)]
pub use engine::set_hidden_set_constraints_enabled;
pub use terms::HalfConstraint;
