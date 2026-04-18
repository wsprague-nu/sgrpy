/// Instruction set struct
#[derive(Clone, Debug)]
pub struct MoveList(pub Vec<usize>);

impl MoveList {
    pub fn new() -> MoveList {
        MoveList(Vec::new())
    }
}
