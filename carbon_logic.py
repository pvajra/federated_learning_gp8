import numpy as np

class CarbonGridSimulator:
    def __init__(self):
        self.base_intensities = [100, 350, 600]
        # Cache realized intensities per (zone_id, current_round) to ensure
        # deterministic values within a single simulation round.
        self._intensity_cache = {}

    def get_carbon_intensity(self, zone_id, current_round):
        key = (zone_id, current_round)
        if key in self._intensity_cache:
            return self._intensity_cache[key]

        base = self.base_intensities[zone_id % 3]
        intensity = max(
            20,
            base + (50 * np.sin(current_round / 10.0)) + np.random.normal(0, 10),
        )
        self._intensity_cache[key] = intensity
        return intensity
