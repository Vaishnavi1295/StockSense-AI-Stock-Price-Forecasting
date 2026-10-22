from src.evaluate import calculate_metrics, directional_accuracy

def test_metrics_exact():
    result = calculate_metrics([1, 2, 3], [1, 2, 4])
    assert round(result["MAE"], 4) == 0.3333
    assert round(result["RMSE"], 4) == 0.5774

def test_directional_accuracy():
    assert directional_accuracy([10, 11, 10], [10, 12, 11]) == 100.0
