use super::config::Config;
use rand::RngExt;

#[derive(Debug, PartialEq, Clone)]
pub struct Genome {
    genes: Vec<f32>,
}

impl Genome {
    pub fn new(genes: Vec<f32>) -> Self {
        Genome { genes }
    }

    pub fn genes(&self) -> &[f32] {
        &self.genes
    }

    pub fn genes_mut(&mut self) -> &mut [f32]{
        &mut self.genes
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn random<R: rand::Rng>(config: &Config, rng: &mut R) -> Self {
        let length = config.genome_length;
        let min = config.initial_gene_min;
        let max = config.initial_gene_max;

        let mut genes = Vec::new();
        for _ in 0..length {
            let gene = rng.random_range(min..=max);
            genes.push(gene);
        }

        Self::new(genes)
    }
}

#[cfg(test)]
mod tests {
    use crate::genetic::Config;

    use super::Genome;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn genome_stores_provided_genes() {
        let genome = Genome::new(vec![0.5, -0.2, 0.9]);
        assert_eq!(genome.genes(), &vec![0.5, -0.2, 0.9]);
    }

    #[test]
    fn genome_reports_correct_length() {
        let genome = Genome::new(vec![0.5, -0.2, 0.9]);
        assert_eq!(genome.len(), 3);
    }

    #[test]
    fn empty_genome_is_allowed() {
        let genome = Genome::new(vec![]);

        assert_eq!(genome.len(), 0);
        assert_eq!(genome.genes().is_empty(), true);
    }

    #[test]
    fn random_genome_has_configured_length() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let genome = Genome::random(&config, &mut rng);

        assert_eq!(genome.len(), config.genome_length);
    }

    #[test]
    fn random_genes_stay_within_configured_bounds() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);

        let genome = Genome::random(&config, &mut rng);

        assert!(
            genome
                .genes()
                .iter()
                .all(|&gene| gene >= config.initial_gene_min && gene <= config.initial_gene_max)
        );
    }

    #[test]
    fn consecutive_random_genomes_are_different() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);

        let mut sequence_a: Vec<Genome> = Vec::new();
        let mut sequence_b: Vec<Genome> = Vec::new();

        for _ in 0..10 {
            let genome = Genome::random(&config, &mut rng);
            sequence_a.push(genome);
        }

        for _ in 0..10 {
            let genome = Genome::random(&config, &mut rng);
            sequence_b.push(genome);
        }

        assert_ne!(sequence_a, sequence_b);
    }

    #[test]
    fn same_seed_reproduces_same_genome_sequence() {
        let config = Config::default();

        let mut sequence_a: Vec<Genome> = Vec::new();
        let mut sequence_b: Vec<Genome> = Vec::new();

        let mut rng_a = StdRng::seed_from_u64(config.seed);
        for _ in 0..10 {
            let genome = Genome::random(&config, &mut rng_a);
            sequence_a.push(genome);
        }

        let mut rng_b = StdRng::seed_from_u64(config.seed);
        for _ in 0..10 {
            let genome = Genome::random(&config, &mut rng_b);
            sequence_b.push(genome);
        }

        assert_eq!(sequence_a, sequence_b);
    }
}
