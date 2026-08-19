use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Checkpoint {
    pub inner: Vec2,
    pub outer: Vec2,
    pub center: Vec2,
}

#[derive(Resource, Clone, Debug)]
pub struct Track {
    pub outer_wall: Vec<Vec2>,
    pub inner_wall: Vec<Vec2>,
    pub checkpoints: Vec<Checkpoint>,
}

impl Default for Track {
    fn default() -> Self {
        // One intentionally hand-authored circuit. Matching inner/outer indices
        // also define checkpoint gates across the road.
        let outer_wall = vec![
            Vec2::new(-470.0, -305.0),
            Vec2::new(105.0, -320.0),
            Vec2::new(420.0, -230.0),
            Vec2::new(500.0, -35.0),
            Vec2::new(450.0, 185.0),
            Vec2::new(275.0, 320.0),
            Vec2::new(-65.0, 330.0),
            Vec2::new(-250.0, 250.0),
            Vec2::new(-445.0, 270.0),
            Vec2::new(-535.0, 95.0),
            Vec2::new(-515.0, -145.0),
        ];
        let inner_wall = vec![
            Vec2::new(-350.0, -175.0),
            Vec2::new(75.0, -185.0),
            Vec2::new(290.0, -120.0),
            Vec2::new(350.0, 0.0),
            Vec2::new(315.0, 105.0),
            Vec2::new(220.0, 175.0),
            Vec2::new(0.0, 190.0),
            Vec2::new(-135.0, 120.0),
            Vec2::new(-315.0, 140.0),
            Vec2::new(-385.0, 45.0),
            Vec2::new(-380.0, -70.0),
        ];
        let checkpoints = outer_wall
            .iter()
            .zip(inner_wall.iter())
            .map(|(&outer, &inner)| Checkpoint {
                inner,
                outer,
                center: (inner + outer) * 0.5,
            })
            .collect();

        Self {
            outer_wall,
            inner_wall,
            checkpoints,
        }
    }
}

pub fn closed_segments(points: &[Vec2]) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
    points
        .iter()
        .copied()
        .zip(points.iter().copied().cycle().skip(1))
        .take(points.len())
}

pub fn crossed_gate(previous: Vec2, current: Vec2, gate: &Checkpoint) -> bool {
    segments_intersect(previous, current, gate.inner, gate.outer)
}

fn segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let movement = b - a;
    let gate = d - c;
    let denominator = movement.perp_dot(gate);
    if denominator.abs() < 1e-6 {
        return false;
    }
    let offset = c - a;
    let t = offset.perp_dot(gate) / denominator;
    let u = offset.perp_dot(movement) / denominator;
    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> Checkpoint {
        Checkpoint {
            inner: Vec2::new(0.0, -2.0),
            outer: Vec2::new(0.0, 2.0),
            center: Vec2::ZERO,
        }
    }

    #[test]
    fn detects_crossing_through_checkpoint() {
        assert!(crossed_gate(
            Vec2::new(-1.0, 0.0),
            Vec2::new(1.0, 0.0),
            &gate()
        ));
    }

    #[test]
    fn rejects_movement_outside_checkpoint_span() {
        assert!(!crossed_gate(
            Vec2::new(-1.0, 3.0),
            Vec2::new(1.0, 3.0),
            &gate()
        ));
    }
}
