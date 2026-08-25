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
