import numpy as np

class CarbonGridSimulator:
    def __init__(self):
        self.base_intensities = [100, 350, 600]
        self._intensity_cache = {} # Prevent inconsistent values in same round

    def reset_cache(self):
        """Call this to clear memory between full simulation runs."""
        self._intensity_cache.clear()

    def get_carbon_intensity(self, zone_id, current_round):
        key = (zone_id, current_round)
        if key in self._intensity_cache:
            return self._intensity_cache[key]
        
        base = self.base_intensities[zone_id % 3]
        # Stochastic sine wave modeling
        val = max(20, base + (50 * np.sin(current_round / 10.0)) + np.random.normal(0, 10))
        
        self._intensity_cache[key] = val
        return val
