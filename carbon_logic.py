# =============================================================================
# carbon_logic.py
# Stochastic carbon-intensity grid simulator — Animesh Kumar
#
# Simulates three geographic energy zones with sinusoidal daily patterns
# and Gaussian noise, modelling real-world renewable grid fluctuation.
#
# Provides utilities to score clients by carbon intensity in carbon-aware simulations.
# =============================================================================
import math
import random
from typing import Dict

_ZONES: Dict[str, Dict[str, float]] = {
    'green': {'base': 80.0,  'amplitude': 30.0,  'noise_std': 10.0},
    'mixed': {'base': 250.0, 'amplitude': 80.0,  'noise_std': 25.0},
    'coal':  {'base': 550.0, 'amplitude': 120.0, 'noise_std': 40.0},
}
_ZONE_NAMES = list(_ZONES.keys())
_MAX_CARBON = (_ZONES['coal']['base'] + _ZONES['coal']['amplitude']
               + 3.0 * _ZONES['coal']['noise_std'])
_carbon_cache: Dict[int, Dict[str, float]] = {}

def reset_carbon_cache():
    _carbon_cache.clear()

def get_zone_for_index(idx: int, num_clients: int) -> str:
    """Assign client by position-index (0..num_clients-1) to a zone."""
    return _ZONE_NAMES[min(idx // max(num_clients // 3, 1), 2)]

def get_carbon_intensity_by_index(idx: int, round_num: int, num_clients: int) -> float:
    """Return gCO2eq/kWh for a client identified by its position index."""
    if round_num not in _carbon_cache:
        _carbon_cache[round_num] = {}
        for zname, p in _ZONES.items():
            sine = p['amplitude'] * math.sin(2.0 * math.pi * round_num / 24.0)
            _carbon_cache[round_num][zname] = max(
                p['base'] + sine + random.gauss(0, p['noise_std']), 10.0)
    return _carbon_cache[round_num][get_zone_for_index(idx, num_clients)]

def get_normalised_carbon_by_index(idx: int, round_num: int, num_clients: int) -> float:
    return min(get_carbon_intensity_by_index(idx, round_num, num_clients) / _MAX_CARBON, 1.0)

if __name__ == "__main__":
    reset_carbon_cache()
    print('✅ Carbon logic ready')
    print('   Sample intensities at round 5 (gCO2eq/kWh):')
    for i, z in enumerate(_ZONE_NAMES):
        print(f'   {z:>6}: {get_carbon_intensity_by_index(i*17, 5, 50):.1f}')
    reset_carbon_cache()
