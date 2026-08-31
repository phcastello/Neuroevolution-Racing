use super::genome::Genome;

#[derive(Clone)]
pub struct Individual {
    genome: Genome,
    fitness: Option<f32>,
}

impl Individual {
    pub fn new(genome: Genome) -> Self {
        Individual {
            genome,
            fitness: None,
        }
    }

    /// Reconstructs an individual from an application-owned persistence DTO.
    pub fn from_parts(genome: Genome, fitness: Option<f32>) -> Result<Self, &'static str> {
        if fitness.is_some_and(|value| !value.is_finite()) {
            return Err("fitness must be finite when present");
        }
        Ok(Self { genome, fitness })
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }

    pub fn fitness(&self) -> Option<f32> {
        self.fitness
    }

    pub fn set_fitness(&mut self, fitness: f32) {
        self.fitness = Some(fitness);
    }
}

#[cfg(test)]
mod tests {
    use super::{Genome, Individual};

    #[test]
    fn test_new_individual_cration() {
        let individual = Individual::new(Genome::new(vec![1.0, 2.0, 3.0]));

        assert_eq!(individual.fitness, None);
        assert_eq!(individual.genome().genes(), &vec![1.0, 2.0, 3.0]);
        assert_eq!(individual.genome().len(), 3);
    }

    #[test]
    fn test_fitness_attribution() {
        let mut individual = Individual::new(Genome::new(vec![1.0, 2.0, -3.0]));

        individual.set_fitness(12.0);

        assert_eq!(individual.fitness(), Some(12.0));
    }
}
