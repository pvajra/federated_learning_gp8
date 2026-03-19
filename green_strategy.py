import flwr as fl
import numpy as np
from carbon_logic import CarbonGridSimulator
from dqn_agent import DQNAgent

class GreenDRLStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.carbon_sim = CarbonGridSimulator()
        self.agent = DQNAgent()
        self.history = {"carbon": [], "energy": []}
        self.round_carbon = {} # Cache for consistency

    def configure_fit(self, server_round, parameters, client_manager):
        clients = list(client_manager.all().values())
        
        # Logic Fix: Score each client based on their specific carbon zone
        scored = []
        for idx, c in enumerate(clients):
            zone_id = idx % 3
            client_carbon = self.carbon_sim.get_carbon_intensity(zone_id, server_round)
            # State: [Carbon intensity, Battery(fixed), CPU(fixed), Round Progress]
            state = [client_carbon / 600, 0.8, 0.1, server_round / 100]
            score = self.agent.select_client_score(state)
            scored.append((c, score))
            
        # Cache mean carbon for accounting in aggregate_fit
        self.round_carbon[server_round] = np.mean([self.carbon_sim.get_carbon_intensity(i, server_round) for i in range(3)])
        
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in scored[:10]]
        return [(c, fl.common.FitIns(parameters, {})) for c in selected]

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        
        if results:
            # Use .get() to prevent crashes if metrics are missing
            total_e = sum((res.metrics or {}).get("total_energy", 0.0) for _, res in results)
            
            # Use cached carbon value for this round
            curr_c = self.round_carbon.get(server_round, 300.0)
            
            self.history["energy"].append(total_e)
            self.history["carbon"].append(total_e * curr_c)
            
        return agg_params, metrics
