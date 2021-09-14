"""Conduct unit testing for fifeforspark.base_modelers module."""

from fifeforspark import base_modelers
import numpy as np

def test_compute_metrics_for_binary_outcome(fabricate_forecasts):
    """Test that FIFE produces correct example AUROC and confusion matrices."""
    errors_list = []
    metrics = {}
    totals = fabricate_forecasts["AUROC=1"][0].count()

    metrics["AUROC=1"] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=1"][0], 
        fabricate_forecasts["AUROC=1"][1], 
        total = totals
    )
    metrics["AUROC=0"] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=0"][0], 
        fabricate_forecasts["AUROC=0"][1], 
        total = totals
    )
    metrics["empty actual"] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["empty actual"][0], 
        fabricate_forecasts["empty actual"][1], 
        total = totals
    )
    metrics[
        "AUROC=1, threshold_positive=1"
    ] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=1"][0],
        fabricate_forecasts["AUROC=1"][1],
        threshold_positive=1.0,
        total = totals
    )
    metrics[
        "AUROC=1, threshold_positive=predicted"
    ] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=1"][0],
        fabricate_forecasts["AUROC=1"][1],
        threshold_positive='predicted',
        total = totals
    )
    metrics[
        "AUROC=1, share_positive=predicted"
    ] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=1"][0],
        fabricate_forecasts["AUROC=1"][1],
        share_positive='predicted',
        total = totals
    )
        
    if not metrics["AUROC=1"]["AUROC"] == 1.0:
        errors_list.append("Condition 1 failed for AUROC=1.")
    if not metrics["AUROC=0"]["AUROC"] == 0.0:
        errors_list.append("Condition 2 failed for AUROC=0.")
        
    if not np.isnan(metrics["empty actual"]["AUROC"]):
        errors_list.append("Condition 3 failed for empty actual.")
        
    if not metrics["AUROC=1, threshold_positive=1"]["True Positives"] == 0:
        errors_list.append("Condition 4 failed for True Positives.")
    if not metrics["AUROC=1, threshold_positive=1"]["False Negatives"] == 4:
        errors_list.append("Condition 5 failed for False Negatives.")
    if not metrics["AUROC=1, threshold_positive=1"]["True Negatives"] == 5:
        errors_list.append("Condition 6 failed for True Negatives.")
    if not metrics["AUROC=1, threshold_positive=1"]["False Positives"] == 0:
        errors_list.append("Condition 7 failed for False Positives.")
        
    if not metrics["AUROC=1, threshold_positive=predicted"]["AUROC"] == 1.0:
        errors_list.append("Condition 8 failed for AUROC=1, predicted threshold.")
        
    if not metrics["AUROC=1, share_positive=predicted"]["AUROC"] == 1.0:
        errors_list.append("Condition 8 failed for AUROC=1, predicted share.")
        
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


SEED = 9999
np.random.seed(SEED)