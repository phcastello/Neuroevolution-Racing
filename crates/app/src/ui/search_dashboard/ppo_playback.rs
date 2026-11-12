//! Bind one validated candidate to one checkpoint before starting playback.
use super::*;

impl SearchDashboard {
    pub(super) fn begin_selected_ppo_test(&mut self, shell: &mut Shell) {
        let track = self
            .shown_candidate()
            .filter(|c| c.geometrically_valid())
            .and_then(|c| c.candidate.as_ref())
            .cloned();
        let Some(track) = track else {
            self.status = "Selecione uma pista geometricamente válida para testar com PPO".into();
            return;
        };
        if let Err(error) = track.validate() {
            self.status = format!("Pista inválida para teste PPO: {error}");
            return;
        }
        self.ppo_test_track = Some(track);
        if self.ppo_checkpoint.is_empty() {
            self.ppo_checkpoint = if !shell.policy_path.is_empty() {
                shell.policy_path.clone()
            } else {
                self.checkpoint.clone()
            };
        }
        self.queue_ppo_test(shell);
    }
    fn queue_ppo_test(&mut self, shell: &mut Shell) {
        shell.status.clear();
        if !PathBuf::from(self.ppo_checkpoint.trim()).is_file() {
            self.status = "Selecione abaixo um checkpoint PPO para testar esta pista".into();
            return;
        }
        let Some(track) = &self.ppo_test_track else {
            return;
        };
        if let Err(error) = track.validate() {
            self.status = format!("Pista inválida para teste PPO: {error}");
            return;
        }
        self.pending_track = Some(track.definition.clone());
        shell.track = track.definition.id.clone();
        shell.policy_path = self.ppo_checkpoint.trim().into();
        shell.cars = shell.cars.clamp(1, 16);
        shell.load_policy = true;
        self.status = "Abrindo o checkpoint PPO na pista selecionada...".into();
    }
    pub(super) fn playback_controls(&mut self, ui: &mut egui::Ui, shell: &mut Shell) {
        let Some(track) = &self.ppo_test_track else {
            return;
        };
        let label = format!("Pista do teste: {}", track.definition.id);
        ui.group(|ui| {
            ui.strong("Testar pista com PPO");
            ui.label(label);
            ui.label("Escolha o checkpoint para esta pista. O teste não treina a política.");
            let chosen = ui
                .push_id("aco_playback_policy_picker", |ui| {
                    super::super::ppo_checkpoint_browser::draw_policy_picker(
                        ui,
                        &mut shell.checkpoint_browser,
                        &mut self.ppo_checkpoint,
                    )
                })
                .inner;
            ui.add(egui::Slider::new(&mut shell.cars, 1..=16).text("Veículos"));
            let start = self.button(ui, "Iniciar teste PPO");
            let cancel = self.button(ui, "Cancelar teste PPO");
            if cancel {
                self.ppo_test_track = None;
            } else if start || chosen {
                self.queue_ppo_test(shell);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_choice_keeps_the_candidate_selected_before_live_search_moves_on() {
        let config = procedural::GeneratorConfig {
            radial_levels: vec![1.],
            ..procedural::GeneratorConfig::legacy()
        };
        let track = procedural::from_decisions(&config, 47, 0, &vec![0; 13]).unwrap();
        let expected = track.definition.clone();
        let record = AntSnapshot {
            ant: 0,
            decisions: vec![],
            candidate: Some(track),
            rejection: None,
            fitness: None,
            rejection_reason: None,
            evaluation: None,
            geometry_valid: Some(true),
            diagnostic: None,
            retention_evicted: false,
            candidate_id: Some(expected.id.clone()),
        };
        let mut d = SearchDashboard {
            config: AcoConfig {
                generator: config,
                ..Default::default()
            },
            current: Some(record),
            ..Default::default()
        };
        let mut shell = Shell::default();
        d.begin_selected_ppo_test(&mut shell);
        assert!(d.ppo_test_track.is_some());
        assert!(d.pending_track.is_none());
        assert!(!shell.load_policy);
        d.receive_event(SearchEvent::AntStarted {
            iteration: 1,
            ant: 1,
            pheromones: vec![],
        });
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../target/ppo-preview-queue-{}.ron",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(
            &path,
            "queue fixture; actual policy restore is tested by the viewer integration",
        )
        .unwrap();
        d.ppo_checkpoint = path.display().to_string();
        d.queue_ppo_test(&mut shell);
        assert!(shell.load_policy);
        assert_eq!(shell.track, expected.id);
        assert_eq!(d.pending_track.as_ref(), Some(&expected));
        assert_eq!(shell.policy_path, path.display().to_string());
        std::fs::remove_file(path).unwrap();
    }
}
