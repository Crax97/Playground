use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

#[derive(
    Default, Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Tick(u128);

static START_NOW: OnceLock<std::time::SystemTime> = std::sync::OnceLock::new();

impl Tick {
    pub const ZERO: Tick = Tick(0);
    pub const MAX: Tick = Tick(u128::MAX);
    pub fn now() -> Self {
        let start_now = START_NOW.get_or_init(std::time::SystemTime::now);
        let now = std::time::SystemTime::now();
        let delta = now
            .duration_since(*start_now)
            .expect("Failed to compute now");
        Self(delta.as_millis())
    }
}
