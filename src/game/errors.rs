#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    IllegalAction,
    CannotUndo,
}
