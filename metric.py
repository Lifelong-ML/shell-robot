class Metric:

    def __init__(self, loss : float, pixel_average_accuracy: float,
                 calibration_error: float):
        assert 0 <= loss, f"loss must be non-negative. Got {loss}"
        assert 0 <= pixel_average_accuracy <= 1, f"pixel_average_accuracy must be between 0 and 1. Got {pixel_average_accuracy}"
        assert 0 <= calibration_error <= 1, f"calibration_error must be between 0 and 1. Got {calibration_error}"
        self.loss = loss
        self.pixel_average_accuracy = pixel_average_accuracy
        self.calibration_error = calibration_error