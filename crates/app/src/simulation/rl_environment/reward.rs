//! Dense GA-inspired objective; see docs/rl/PPO_REWARD_GA_INSPIRED.txt.
//! Only the existing validated high-water gain enters this calculation.
use racing_rl::config::{RewardConfig, RewardMode};

pub(super) fn progress_reward(config: &RewardConfig, gain: f32, length: f32, dt: f32) -> f32 {
    match config.mode {
        // Preserve the original arithmetic, units and scale for historical runs.
        RewardMode::Baseline => gain * config.progress_weight,
        RewardMode::GaInspired {
            speed_weight,
            progress_speed_half_saturation,
        } => {
            if gain <= 0. {
                return 0.;
            }
            let useful_speed = gain / dt;
            let speed_factor = super::super::training::normalize_useful_progress_speed(
                useful_speed,
                progress_speed_half_saturation,
            );
            (gain / length) * (config.progress_weight + speed_weight * speed_factor)
        }
    }
}
