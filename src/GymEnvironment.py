import gym
import numpy as np
from gym import spaces
from config_RL import *
from environment_SimPy import *


class InventoryManagementEnv(gym.Env):
    def __init__(self):
        super(InventoryManagementEnv, self).__init__()
        self.action_space = ACTION_SPACE_SIZE
        self.observation_space = MULTI_STATE_SPACE_SIZE
        self.daily_events = []
        self.simpy_env, self.inventory_list, self.procurement_list, self.production_list, \
            self.sales, self.customer, self.supplier_list, self.daily_events = create_env(
                I, P, self.daily_events)
        self.current_step = 0

    def reset(self):
        self.daily_events = []
        self.simpy_env, self.inventory_list, self.procurement_list, self.production_list, \
            self.sales, self.customer, self.supplier_list, self.daily_events = create_env(
                I, P, self.daily_events)
        simpy_event_processes(self.simpy_env, self.inventory_list, self.procurement_list, self.production_list,
                              self.sales, self.customer, self.supplier_list, self.daily_events, I,
                              {"DEMAND": DEMAND_SCENARIO, "LEADTIME": LEADTIME_SCENARIO})
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # Apply action (order quantities to suppliers)
        for i, qty in enumerate(action):
            self.procurement_list[i].order_qty = qty
            if qty > 0:
                self.inventory_list[self.supplier_list[i].item_id].update_inven_level(
                    qty, "IN_TRANSIT", self.daily_events)

        # Simulate one day
        self.simpy_env.run(until=self.simpy_env.now + 24)
        self.current_step += 1

        # Calculate reward
        cost = Cost.update_cost_log(self.inventory_list)
        reward = -cost

        # Update daily report
        update_daily_report(self.inventory_list)

        # Get next state
        state = self._get_state()

        # Check if done
        done = self.current_step >= MAX_STEPS

        return state, reward, done, {}

    def _get_state(self):
        state = []
        # On-hand inventory levels
        for inv in self.inventory_list:
            state.append(inv.on_hand_inventory)
        # In-transition inventory levels for materials
        for inv in self.inventory_list:
            if I[inv.item_id]["TYPE"] == "Material":
                state.append(inv.in_transition_inventory)
        # Remaining demand
        state.append(I[0]["DEMAND_QUANTITY"])
        state = np.array(state, dtype=np.float32)
        # Normalize state to [0, 1]
        normalized_state = (state - STATE_MINS) / \
            (STATE_MAXS - STATE_MINS + 1e-8)
        return normalized_state

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        for inv in self.inventory_list:
            print(
                f"{I[inv.item_id]['NAME']}: On-hand={inv.on_hand_inventory}, In-transit={inv.in_transition_inventory}")

    def close(self):
        pass

# import gym
# from gym import spaces
# import numpy as np
# from typing import Tuple, Dict, Any, List

# from config_SimPy import *
# from config_RL import *
# from environment_SimPy import *


# class InventoryManagementEnv(gym.Env):
#     """
#     A Gym environment for inventory management using SimPy.
#     """

#     def __init__(self):
#         """
#         Initialize the environment.
#         """
#         super(InventoryManagementEnv, self).__init__()

#         # Define action and observation spaces
#         self.action_space = ACTION_SPACE_SIZE
#         self.observation_space = MULTI_STATE_SPACE_SIZE

#         # Initialize environment components
#         self.env = None
#         self.inventoryList = []
#         self.procurementList = []
#         self.productionList = []
#         self.sales = None
#         self.customer = None
#         self.supplierList = []
#         self.daily_events = []

#         # Initialize state
#         self.current_state = None
#         self.day = 0
#         self.total_reward = 0
#         self.done = False
#         self.info = {}

#     def reset(self):
#         """
#         Reset the environment to initial state.

#         Returns:
#             np.array: Initial state
#         """
#         # Reset simulation time
#         self.day = 0
#         self.total_reward = 0
#         self.done = False

#         # Clear daily events
#         self.daily_events = []

#         # Create environment with SimPy
#         self.env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.supplierList, self.daily_events = create_env(
#             I, P, self.daily_events)

#         # Setup scenario
#         scenario = {
#             "DEMAND": DEMAND_SCENARIO,
#             "LEADTIME": LEADTIME_SCENARIO
#         }

#         # Set up processes
#         simpy_event_processes(self.env, self.inventoryList, self.procurementList, self.productionList,
#                               self.sales, self.customer, self.supplierList, self.daily_events, I, scenario)

#         # Get initial state
#         self.current_state = self._get_state()

#         # Clear costs
#         Cost.clear_cost()

#         return self.normalize_state(self.current_state)

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
#         """
#         Take a step in the environment.

#         Args:
#             action (np.ndarray): Order quantities for materials

#         Returns:
#             Tuple[np.ndarray, float, bool, Dict[str, Any]]: Next state, reward, done flag, info
#         """
#         # Set the order quantities for each material
#         mat_idx = 0
#         for i in I.keys():
#             if I[i]["TYPE"] == "Material":
#                 I[i]["LOT_SIZE_ORDER"] = action[mat_idx]
#                 mat_idx += 1

#         # Run simulation for one day
#         self.env.run(until=(self.day + 1) * 24)
#         self.day += 1

#         # Update daily report
#         update_daily_report(self.inventoryList)

#         # Calculate reward (negative cost)
#         daily_cost = Cost.update_cost_log(self.inventoryList)
#         reward = -daily_cost
#         self.total_reward += reward

#         # Clear daily cost for next step
#         Cost.clear_cost()

#         # Check if simulation is done
#         self.done = self.day >= MAX_STEPS_PER_EPISODE

#         # Get next state
#         next_state = self._get_state()
#         self.current_state = next_state

#         # Update info dictionary
#         self.info = {
#             "day": self.day,
#             "daily_cost": daily_cost,
#             "total_reward": self.total_reward,
#             "inventory_levels": {I[inven.item_id]['NAME']: inven.on_hand_inventory for inven in self.inventoryList},
#             "in_transit": {I[inven.item_id]['NAME']: inven.in_transition_inventory
#                            for inven in self.inventoryList if I[inven.item_id]['TYPE'] == 'Material'}
#         }

#         return self.normalize_state(next_state), reward, self.done, self.info

#     def render(self, mode='human'):
#         """
#         Render the environment.

#         Args:
#             mode (str): Rendering mode
#         """
#         if mode == 'human':
#             print(f"Day: {self.day}")
#             print(f"State: {self.current_state}")
#             print(f"Total Reward: {self.total_reward}")

#             # Print inventory levels
#             print("Inventory Levels:")
#             for inven in self.inventoryList:
#                 print(
#                     f"  {I[inven.item_id]['NAME']}: {inven.on_hand_inventory}")
#                 if I[inven.item_id]['TYPE'] == 'Material':
#                     print(
#                         f"  {I[inven.item_id]['NAME']} In-Transit: {inven.in_transition_inventory}")

#             print("------------------------------")

#     def close(self):
#         """
#         Clean up resources.
#         """
#         pass

#     def _get_state(self) -> np.ndarray:
#         """
#         Get the current state of the environment.

#         Returns:
#             np.ndarray: Current state
#         """
#         state = []

#         # On-hand inventory levels for all items
#         for inven in self.inventoryList:
#             state.append(inven.on_hand_inventory)

#         # In-transition inventory levels for material items
#         for inven in self.inventoryList:
#             if I[inven.item_id]['TYPE'] == 'Material':
#                 state.append(inven.in_transition_inventory)

#         # Remaining demand
#         state.append(I[0]["DEMAND_QUANTITY"])

#         return np.array(state, dtype=np.int32)

#     def normalize_state(self, state: np.ndarray) -> np.ndarray:
#         """
#         Normalize state to range [0, 1].

#         Args:
#             state (np.ndarray): State to normalize

#         Returns:
#             np.ndarray: Normalized state
#         """
#         return (state - STATE_MINS) / (STATE_MAXS - STATE_MINS)
