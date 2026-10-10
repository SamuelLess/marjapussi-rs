//! Compile-time storage choices; game rules do not depend on the public API feature.
use crate::game::{cards::Card, gameinfo::GameMetaInfo};
#[cfg(feature = "public-api")]
pub type List<T, const N: usize> = Vec<T>;
#[cfg(not(feature = "public-api"))]
pub type List<T, const N: usize> = crate::inline::InlineVec<T, N>;
#[cfg(feature = "public-api")]
pub type Name = String;
#[cfg(not(feature = "public-api"))]
pub type Name = std::sync::Arc<str>;
#[cfg(feature = "public-api")]
pub type Meta = GameMetaInfo;
#[cfg(not(feature = "public-api"))]
pub type Meta = std::sync::Arc<GameMetaInfo>;
#[cfg(feature = "public-api")]
pub type Pass = Vec<Card>;
#[cfg(not(feature = "public-api"))]
pub type Pass = [Card; 4];
#[cfg(feature = "public-api")]
pub type Time = String;
#[cfg(not(feature = "public-api"))]
pub type Time = u64;

#[cfg(not(feature = "public-api"))]
mod fillers {
    use crate::{
        game::{cards::Card, gamestate::FinishedTrick, player::PlaceAtTable, points::Points},
        inline::Filler,
    };
    impl Filler for PlaceAtTable {
        const FILL: Self = Self(0);
    }
    impl Filler for FinishedTrick {
        const FILL: Self = Self {
            cards: [Card::FILL; 4],
            winner: PlaceAtTable(0),
            points: Points(0),
        };
    }
}
