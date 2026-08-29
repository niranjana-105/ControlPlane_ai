from controlplane.benchmark_scenarios import BenchmarkRunner
from controlplane.config import PolicyProfileType

runner = BenchmarkRunner()
results = runner.run_all(
    nli_threshold=0.65,
    bias_threshold=0.35,
    entropy_threshold=2.5,
    profile_type=PolicyProfileType.CUSTOMER_SUPPORT,
)
stats = runner.aggregate_stats(results)
print("Pass Rate:", round(stats["pass_rate"]*100, 1), "%")
print("Avg F1:   ", round(stats["avg_f1"]*100, 1), "%")
print("Avg Prec: ", round(stats["avg_precision"]*100, 1), "%")
print("Avg Rec:  ", round(stats["avg_recall"]*100, 1), "%")
print()
for cat, data in stats["by_category"].items():
    print("  " + cat + ": Pass=" + str(round(data["pass_rate"]*100)) + "%, F1=" + str(round(data["avg_f1"]*100)) + "%")
