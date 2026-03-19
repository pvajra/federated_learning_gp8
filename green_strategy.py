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
        # Cache carbon intensity per round to ensure consistent use between
        # client selection (configure_fit) and accounting (aggregate_fit)
        self.round_carbon = {}

    def configure_fit(self, server_round, parameters, client_manager):
        clients = list(client_manager.all().values())
        curr_carbon = np.mean([self.carbon_sim.get_carbon_intensity(i, server_round) for i in range(3)])
        # Cache the carbon intensity for this round so aggregate_fit can use
        # the same value for accounting
        self.round_carbon[server_round] = curr_carbon
        scored = [
            (c, self.agent.select_client_score([curr_carbon / 600, 0.8, 0.1, server_round / 100]))
            for c in clients
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in scored[:10]]
        return [(c, fl.common.FitIns(parameters, {})) for c in selected]

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if results:
            total_e = sum([res.metrics["total_energy"] for _, res in results])
            # Reuse cached carbon intensity for this round if available to
            # ensure consistency with configure_fit; otherwise compute once.
            curr_c = self.round_carbon.get(server_round)
            if curr_c is None:
                curr_c = np.mean(
                    [self.carbon_sim.get_carbon_intensity(i, server_round) for i in range(3)]
                )
                self.round_carbon[server_round] = curr_c
            self.history["energy"].append(total_e)
            self.history["carbon"].append(total_e * curr_c)
        return agg_params, metrics
