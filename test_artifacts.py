
from src.services.artifact_service import (
    get_experiment_report,
    get_candidate_results,
    get_q_table,
    get_candidate_configs,
    get_search_history
)

report = get_experiment_report()
candidates = get_candidate_results()
q_table = get_q_table()
configs = get_candidate_configs()
history = get_search_history()

print("========== ARTIFACT SERVICE TEST ==========")
print("Report loaded:", bool(report))
print("Candidates:", len(candidates))
print("Q-table shape:", q_table.shape)
print("Configurations:", len(configs))
print("Search history entries:", len(history))
print("\nBest candidate:")
print(candidates.loc[candidates["reward"].idxmax()])
print("\nAll artifact services working.")
