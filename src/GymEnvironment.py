import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any, List

from config_SimPy import *
from config_RL import *
from environment_SimPy import *


class InventoryManagementEnv(gym.Env):
    """
    A Gym environment for inventory management using SimPy.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        super(InventoryManagementEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = ACTION_SPACE_SIZE
        self.observation_space = MULTI_STATE_SPACE_SIZE

        # Initialize environment components
        self.env = None
        self.inventoryList = []
        self.procurementList = []
        self.productionList = []
        self.sales = None
        self.customer = None
        self.supplierList = []
        self.daily_events = []

        # Initialize state
        self.current_state = None
        self.day = 0
        self.total_reward = 0
        self.done = False
        self.info = {}

    def reset(self):
        """
        Reset the environment to initial state.

        Returns:
            np.array: Initial state
        """
        # Reset simulation time
        self.day = 0
        self.total_reward = 0
        self.done = False

        # Clear daily events
        self.daily_events = []

        # Create environment with SimPy
        self.env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.supplierList, self.daily_events = create_env(
            I, P, self.daily_events)

        # Setup scenario
        scenario = {
            "DEMAND": DEMAND_SCENARIO,
            "LEADTIME": LEADTIME_SCENARIO
        }

        # Set up processes with explicit simulation time limit
        simpy_event_processes(self.env, self.inventoryList, self.procurementList, self.productionList,
                              self.sales, self.customer, self.supplierList, self.daily_events, I, scenario)

        # print("Initial simulation run...")
        self.env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.supplierList, self.daily_events = create_env(
            I, P, self.daily_events)
        simpy_event_processes(self.env, self.inventoryList, self.procurementList, self.productionList,
                              self.sales, self.customer, self.supplierList, self.daily_events, I, scenario)

        # Get initial state
        self.current_state = self._get_state()

        # Clear costs
        Cost.clear_cost()

        # print("Environment reset complete")
        return self.normalize_state(self.current_state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (np.ndarray): Order quantities for materials

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Next state, reward, done flag, info
        """
        # Verify action shape matches the number of materials
        if len(action) != MAT_COUNT:
            print(
                f"Warning: Action length {len(action)} does not match material count {MAT_COUNT}. Adjusting action.")
            # Create a corrected action of proper length
            corrected_action = np.zeros(MAT_COUNT, dtype=np.int32)
            for i in range(min(len(action), MAT_COUNT)):
                corrected_action[i] = action[i]
            action = corrected_action

        # Set the order quantities for each material
        mat_idx = 0
        for i in I.keys():
            if I[i]["TYPE"] == "Material":
                I[i]["LOT_SIZE_ORDER"] = int(
                    action[mat_idx])  # Ensure it's an integer
                mat_idx += 1

        # Run simulation for one day
        self.env.run(until=(self.day + 1) * 24)
        self.day += 1

        # Update daily report
        update_daily_report(self.inventoryList)

        # Calculate reward (negative cost)
        daily_cost = Cost.update_cost_log(self.inventoryList)
        reward = -daily_cost
        self.total_reward += reward

        # Clear daily cost for next step
        Cost.clear_cost()

        # Check if simulation is done
        self.done = self.day >= MAX_STEPS_PER_EPISODE

        # Get next state
        next_state = self._get_state()
        self.current_state = next_state

        # Update info dictionary
        self.info = {
            "day": self.day,
            "daily_cost": daily_cost,
            "total_reward": self.total_reward,
            "inventory_levels": {I[inven.item_id]['NAME']: inven.on_hand_inventory for inven in self.inventoryList},
            "in_transit": {I[inven.item_id]['NAME']: inven.in_transition_inventory
                           for inven in self.inventoryList if I[inven.item_id]['TYPE'] == 'Material'},
            "actions": {I[i]['NAME']: I[i]["LOT_SIZE_ORDER"] for i in I.keys() if I[i]["TYPE"] == "Material"}
        }

        return self.normalize_state(next_state), reward, self.done, self.info

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            print(f"Day: {self.day}")
            print(f"State: {self.current_state}")
            print(f"Total Reward: {self.total_reward}")

            # Print inventory levels
            print("Inventory Levels:")
            for inven in self.inventoryList:
                print(
                    f"  {I[inven.item_id]['NAME']}: {inven.on_hand_inventory}")
                if I[inven.item_id]['TYPE'] == 'Material':
                    print(
                        f"  {I[inven.item_id]['NAME']} In-Transit: {inven.in_transition_inventory}")

            print("------------------------------")

    def close(self):
        """
        Clean up resources.
        """
        pass

    def _get_state(self) -> np.ndarray:
        """
        Get the current state of the environment.

        Returns:
            np.ndarray: Current state
        """
        state = []

        # On-hand inventory levels for all items
        for inven in self.inventoryList:
            state.append(inven.on_hand_inventory)

        # In-transition inventory levels for material items
        for inven in self.inventoryList:
            if I[inven.item_id]['TYPE'] == 'Material':
                state.append(inven.in_transition_inventory)

        # Remaining demand
        state.append(I[0]["DEMAND_QUANTITY"])

        return np.array(state, dtype=np.int32)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state to range [0, 1].

        Args:
            state (np.ndarray): State to normalize

        Returns:
            np.ndarray: Normalized state
        """
        return (state - STATE_MINS) / (STATE_MAXS - STATE_MINS)
