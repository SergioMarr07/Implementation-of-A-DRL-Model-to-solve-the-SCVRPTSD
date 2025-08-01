import torch
import numpy as np

from CVRPEnvInference import CVRPEnv as Env
from CVRPModelInference import CVRPModel as Model

from utilsInference import *

# FUNCTIONS FOR INFERENCE
##################################################

'Function to get best metrics from a set of solutions'
# We need to analize this part, every model uses a different configuration to compute its reward during training so it is logic to use the same configuration during inference
def get_best_metrics(state, reward: torch.tensor):
  # identify the index of the best solution from the reward tensor
  idx_max = torch.argmax(reward, dim = -1)

  # index best_metrics for the first instance in the batch
  best_reward = reward[0, idx_max]
  best_solution = state.H_cal[0, idx_max]
  best_t_travel = state.t_travel[0, idx_max]
  best_t_out = state.t_out[0, idx_max]

  return best_reward, best_solution, best_t_travel, best_t_out

'Function to plot ONE graph and its solution for a variant of the CVRP'
def plot_graph(instance_info: tuple,
               solution: np.ndarray,
               accumulated_delays: float,
               show_plot: bool = True,
               show_node_features: bool = True,
               show_legend: bool = True,
               save: bool = False):
    # Extract features of the instance
    coordinates, demands, deadlines, service_times, random_multipliers = instance_info
    # Trim solution to keep one zero at the end
    last_non_zero_idx = np.nonzero(solution)[0][-1]
    solution = np.concatenate((solution[:last_non_zero_idx + 1], [0]))

    # Build full node features array
    node_features = np.concatenate((coordinates,
                                    demands[:, None],
                                    deadlines[:, None],
                                    service_times[:, None]), axis=-1)

    # Plot nodes
    for idx, (x, y, d, dl, st) in enumerate(node_features):
        color = "blue" if idx == 0 else "red"
        plt.scatter(x, y, c=color, label="Depot" if idx == 0 else "")
        label = f"({idx}, {d:.0f}, {dl:.2f}, {st:.2f})" if show_node_features else str(idx)
        plt.text(x + 0.05, y + 0.05, label, size='x-small', stretch='ultra-condensed', ha='left')
    plt.scatter([], [], c="red", label="Delivery location")  # dummy for legend

    # Define route colors
    colors_dic = {
        1: 'k', 2: 'crimson', 3: 'darkorange', 4: 'limegreen', 5: 'yellow',
        6: 'c', 7: 'darkviolet', 8: 'magenta', 9: 'gray', 10: 'saddlebrown',
        11: 'palegreen', 12: 'steelblue', 13: 'violet'
    }

    route_idx, demands_route, t_route, t_travel = 0, 0, 0, 0
    legend_entries = []

    # Loop over solution
    for i, idx in enumerate(solution[:-1]):
        next_idx = solution[i + 1] # Determine the next location to visit in the solution

        if idx == 0: # if the depot is visited
          # Restart some variables of the route
          route = [idx.item()] # creates a route list where the visited locations of the route will be saved
          demands_route, t_route = 0, 0 # restart demands and times of the route
          route_idx += 1 # update idx of the route
          color = colors_dic.get(route_idx, 'black') # Update the color of the route, if the route idx is not in the dictionary then the color used will be black
        else:
          route.append(idx.item()) # add the visited location to the route list
          demands_route += node_features[idx, 2] # update demands of the route

        # Compute edge distances and times
        start_x, start_y = node_features[idx, :2] # filter coordinates of initial nodes
        end_x, end_y = node_features[next_idx, :2] # filter coordinates of final nodes
        edge_dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2) # compute 2D Euclidian distance
        edge_time = random_multipliers[idx, next_idx]*edge_dist # compute edge time
        t_route += (node_features[idx, -1] + edge_time) # update route time
        t_travel += (node_features[idx, -1] + edge_time) # update travel time

        # Plot edges
        plt.arrow(start_x, start_y,
                  end_x - start_x, end_y - start_y,
                  color=color, head_width=0.1, head_length=0.1,
                  length_includes_head=True, lw=2)

        if next_idx == 0:
          route.append(next_idx.item()) # add the final location of the route
          route_info = (
              f"Route {route_idx}: {route}\n"
              f"    Time: {t_route:.2f} min\n"
              f"    Supplies: {int(demands_route)} units"
          )
          legend_entries.append(route_info)

    # Add legend if enabled
    if show_legend:
        plt.plot([], [], color='white', label=f"Total travel time: {t_travel / 60:.2f} h")
        plt.plot([], [], color='white', label=f"Accumulated delays: {accumulated_delays / 60:.2f} h")
        plt.plot([], [], color='white', label=f"Total delivered supplies: {int(demands.sum())} units")
        for idx, entry in enumerate(legend_entries):
            plt.plot([], [], color=colors_dic.get(idx + 1, 'black'), label=entry)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.title(f"Solution: {solution.tolist()}", fontsize=10)

    if save:
        file_name = f"graph_{'_'.join(map(str, solution))}.png"
        plt.savefig(file_name, dpi=240, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return t_travel, accumulated_delays

"Function to compute a solution of a specific instance (fixed seed) with a specific NN-model and optionally plotting it"
def compute_solution(n: int, # number of customers (locations without counting the depot)
                     SEED: int = 11,
                     model_name: str = None,
                     return_features: bool = False,
                     show_plot: bool = True,
                     show_node_features: bool = False,
                     show_legend: bool = True,
                     save: bool = False):
  # DEFINING PARAMETERS FOR ENV AND NN-MODEL GENERATION
  ##################################################

  # Set seeds for random number generation
  torch.manual_seed(SEED)
  if torch.cuda.is_available(): # If using GPU, set the seed for CUDA as well
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU environments

  # Set params for environment and model
  env_params = {
      'problem_size': n,
      'pomo_size': n,
      'used_saved_problems': False}
  model_params = {
      'embedding_dim': 128,
      'sqrt_embedding_dim': 128**(1/2),
      'encoder_layer_num': 6,
      'qkv_dim': 16,
      'head_num': 8,
      'logit_clipping': 10,
      'ff_hidden_dim': 512,
      'eval_type': 'argmax'}
  tester_params = {
      'model_name': model_name,  # epoch version of pre-trained model to load
      'test_episodes': 1,
      'test_batch_size': 1} # number of samples to process
  # Set device
  USE_CUDA = False
  if USE_CUDA:
    cuda_device_num = 0
    torch.cuda.set_device(cuda_device_num)
    device = torch.device('cuda', cuda_device_num)
  else:
    device = torch.device('cpu')

  # Set env and NN-model
  env = Env(**env_params)
  model = Model(**model_params)

  # Restore
  if tester_params['model_name'] is not None:
    model_name = tester_params['model_name']
    checkpoint_path = f'/content/{model_name}'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

  test_num_episode = tester_params['test_episodes']
  episode = 0
  ##################################################

  while episode < test_num_episode:
    remaining = test_num_episode - episode
    batch_size = min(tester_params['test_batch_size'], remaining)

    model.eval()
    with torch.no_grad():
      env.load_problems(batch_size)
      reset_state, _, _ = env.reset()
      model.pre_forward(reset_state)

    # POMO Rollout
    ###############################################
    state, reward, done = env.pre_step()
    while not done:
        a_current, _ = model(state)
        # shape: (batch, pomo)
        state, reward, done = env.step(a_current)

    # Return
    ###############################################

    # Computing with the NN-model
    best_reward, best_solution, best_t_travel, best_t_out = get_best_metrics(state, reward)

    coordinates = env.depot_customers_xy.squeeze().numpy()
    demands = env.depot_customers_demand.squeeze().numpy()
    deadlines = env.depot_customers_deadline.squeeze().numpy()
    service_times = env.depot_customers_service.squeeze().numpy()
    random_multipliers = reset_state.random_multipliers.squeeze().numpy()
    instance_info = (coordinates, demands, deadlines, service_times, random_multipliers)

    t_travel, delays = plot_graph(instance_info = instance_info,
                                  solution = best_solution.squeeze().numpy(),
                                  accumulated_delays = best_t_out.item(),
                                  show_plot = show_plot,
                                  show_node_features = show_node_features,
                                  show_legend = show_legend,
                                  save = save) # to know if you want to save the plot

    episode += batch_size

    if return_features:
      return t_travel, delays, instance_info
    else:
      return t_travel, delays

"Block of code to manage the installation of OR-tools"
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    print("Module ortools is installed")
except ModuleNotFoundError:
    print("Module ortools is not installed, installing it...")
    !pip install ortools
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

"Function to compute a solution of a specific instance (fixed seed) with OR-tools and optionally plotting it"
def compute_solution_with_ortools(instance_info: tuple,
                                  SEED: int = 0,
                                  metaheuristic: str = "GUIDED_LOCAL_SEARCH",
                                  show_plot=False,
                                  show_node_features=False,
                                  show_legend=False,
                                  save=False,
                                  capacity=30):
    # Extract features of the instance
    coordinates, demands, deadlines, service_times, random_multipliers = instance_info
    deadlines[0] = float(99999) # using np.inf as boundary generates conflict with OR-tools

    N = coordinates.shape[0]  # number of nodes (1 depot + customers)
    depot_index = 0
    np.random.seed(SEED)

    # Build time and distance matrices
    distance_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))

    # Assign values to the distance and time matrix based on the coordinates of the nodes or locations
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distance_matrix[i, j] = dist
            time_matrix[i, j] = random_multipliers[i, j] * dist

    # OR-Tools Setup, pywrapcp is the Python wrapper for the C++ constraint programming solver provided by OR-Tools
    manager = pywrapcp.RoutingIndexManager(N, N, depot_index) # this object keep tracks of your problem indices an those used (abstract indices) by the OR-tool solver (they are not the same)
    # Sintax: RoutingIndexManager(number_of_locations, number_of_vehicles, depot_index), the number of vehicles in the fleet has been fixed as N to simulate depot returns
    routing = pywrapcp.RoutingModel(manager) # RoutingModel is the core optimization engine of the library, this object will be used to define the VRP, its constraints, objective and compute the solution

    # Callback for the cost of the edge (edge travel time + service time). A callback, in this case, is a function that will compute and return useful information from and to the OR-tool model
    def time_callback(from_index, to_index): # the arguments are the abstract indices used by OR-tools
        # Convert the abstract indices to user defined indices
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node] + service_times[from_node]) # look for the time elements and return the values, this must be a integer

    transit_callback_index = routing.RegisterTransitCallback(time_callback) # Give the cost edge callback to the routing model, literally you're saying the model: This is how you compute the cost or travel time between two nodes
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index) # set the object from above as the way to compute the cost function

    # Capacity dimension
    demand_callback_index = routing.RegisterUnaryTransitCallback(lambda from_index: -int(demands[manager.IndexToNode(from_index)])) # this defines a callback function that returns the demand at a given node. OR-Tools will use this function to track vehicle load as it moves from node to node
    routing.AddDimensionWithVehicleCapacity( # adds a new constraint to the problem — the Capacity dimension — which tracks the cumulative load on each vehicle.
        demand_callback_index,               # callback for delivery amounts
        0,                                   # no slack (no overcapacity allowed), this is actually the hard capacity constraint
        [capacity] * N,                      # one vehicle per route (each with same max capacity)
        False,                               # fix_start_cumul_to_zero = False
        "Capacity")                          # name of the dimension

    # Time dimension
    routing.AddDimension(
        transit_callback_index, # callback that returns travel time (or time cost) between nodes and its time service
        99999,                  # max "waiting time" allowed at nodes (there are no restricttions)
        99999,                  # max cumulative time per vehicle route (there are no restricttions)
        False,                  # whether vehicles must start at time 0
        "Time")                 # string name of this dimension

    # Add soft time windows (deadline) constraint
    time_dim = routing.GetDimensionOrDie("Time") # access the time dimension defined above
    penalty_per_minute = 20
    for i, deadline in enumerate(deadlines):
      index = manager.NodeToIndex(i) # maps from problem indices to OR-tools indices
      time_dim.SetCumulVarSoftUpperBound(index, int(deadline), penalty_per_minute)

    # Fix the start of the first trip (vehicle 0)
    time_dim.CumulVar(routing.Start(0)).SetValue(0)

    # Sequential dispatching: V1 → V2 → V3 ...
    for v in range(1, N):  # loop over the N virtual vehicles
      routing.SetFixedCostOfVehicle(0, v) # set the cost of using a vehicle of zero, simulating that just one vehicle is used, if first parameter is different from zero, then such a value is added to the cost function

      # Get the indices of the start and end of the routes
      prev_end = routing.End(v - 1) # final index of the previous route (vehicle)
      curr_start = routing.Start(v) # initial index of the current route (vehicle)

      # Get some useful parameters
      prev_end_time = time_dim.CumulVar(prev_end)
      curr_start_time = time_dim.CumulVar(curr_start)

      routing.solver().Add(curr_start_time >= prev_end_time + 15) # add condition that the second vehicle starts after the first one finishes (plus 15 minutes to load the vehicle) and so on

    # Features solver setup
    search_params = pywrapcp.DefaultRoutingSearchParameters() # creates the default search configuration object, below this will be modified
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC # compute initial solution with the Path Cheapest Arc strategy, it greedily picks the next unvisited node with the lowest cost. We could also use other options
    if metaheuristic == "GUIDED_LOCAL_SEARCH":
      search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH # sets the improvement strategy used after the first solution is built, in this case it correspond to Guided Local Search (GLS). We could also use Tabu Search, Simulated Annealing, Generic Tabu Search
    elif metaheuristic == "TABU_SEARCH":
      search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    elif metaheuristic == "SIMULATED_ANNEALING":
      search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    elif metaheuristic == "GENERIC_TABU_SEARCH":
      search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH
    else:
      raise ValueError("Invalid metaheuristic option. Choose from 'GUIDED_LOCAL_SEARCH', 'TABU_SEARCH', 'SIMULATED_ANNEALING', or 'GENERIC_TABU_SEARCH'.")
    search_params.time_limit.seconds = 5 # sets the maximum time (in seconds) the solver is allowed to run

    # Solve
    solution = routing.SolveWithParameters(search_params)

    # Evaluate solution
    ###############################################
    t_delay = 0
    flat_solution = []

    if solution: # if there's a feasible solution for the current instance
      # Get the dimensions of the problem to extract data from that (the time dimension is already called)
      capacity_dim = routing.GetDimensionOrDie("Capacity")

      for v in range(N): # loop over the routes
        index = routing.Start(v) # get the first index of the route
        end_index = routing.End(v) # get the last index of the route

        # Access the cumulative time (arrival time) for start and end
        start_time = solution.Min(time_dim.CumulVar(index))
        end_time = solution.Min(time_dim.CumulVar(end_index))

        if routing.IsEnd(solution.Value(routing.NextVar(index))): # evaluates if the route is "empty"
          continue  # Skip unused vehicle by getting out of the loop


        while not routing.IsEnd(index): # loop over the route
          node_index = manager.IndexToNode(index) # get problem index from OR-tools index
          flat_solution.append(node_index) # save index in flat solution

          load = solution.Value(capacity_dim.CumulVar(index))
          arrival = solution.Value(time_dim.CumulVar(index))

          # Update t_route and t_delay
          if arrival > deadlines[node_index]:
            t_delay += (arrival - deadlines[node_index])

          index = solution.Value(routing.NextVar(index))

# COMPUTING SOLUTION FOR DIFFERENT MODELS
##################################################

from collections import defaultdict
import numpy as np

# Store results in dictionaries for clarity
results = defaultdict(lambda: {'travel_times': [], 'outs': []})
show_plot = False

# Change next statements to evaluate other instance sizes
n = 50
model_name = "results/checkpoint_n50-50.pt"

for seed in range(100):
  # Not trained
  t_travel, t_out, instance_info = compute_solution(n=n, SEED = seed, return_features=True, show_plot=show_plot)
  results['Untrained']['travel_times'].append(t_travel)
  results['Untrained']['outs'].append(t_out)

  # Trained unscaled
  ##################################################
  t_travel, t_out = compute_solution(n=n, SEED = seed, model_name = model_name, show_plot=show_plot)
  results['POMO-DC']['travel_times'].append(t_travel)
  results['POMO-DC']['outs'].append(t_out)

  # OR-Tools using Guided Local Search
  ##################################################
  t_travel, t_out = compute_solution_with_ortools(instance_info=instance_info, SEED=seed, show_plot=show_plot, show_legend = True)
  results['OR-Tools-GLS']['travel_times'].append(t_travel)
  results['OR-Tools-GLS']['outs'].append(t_out)

  # OR-Tools using Tabu Search
  ##################################################
  t_travel, t_out = compute_solution_with_ortools(instance_info=instance_info, SEED=seed, metaheuristic="TABU_SEARCH" , show_plot=show_plot, show_legend = True)
  results['OR-Tools-TS']['travel_times'].append(t_travel)
  results['OR-Tools-TS']['outs'].append(t_out)

  # OR-Tools using Simulated Annealing
  ##################################################
  t_travel, t_out = compute_solution_with_ortools(instance_info=instance_info, SEED=seed, metaheuristic="SIMULATED_ANNEALING" , show_plot=show_plot, show_legend = True)
  results['OR-Tools-SA']['travel_times'].append(t_travel)
  results['OR-Tools-SA']['outs'].append(t_out)

  # OR-Tools using Generic Tabu Search
  ##################################################
  t_travel, t_out = compute_solution_with_ortools(instance_info=instance_info, SEED=seed, metaheuristic="GENERIC_TABU_SEARCH" , show_plot=show_plot, show_legend = True)
  results['OR-Tools-GTS']['travel_times'].append(t_travel)
  results['OR-Tools-GTS']['outs'].append(t_out)


      t_travel, t_delay = plot_graph(instance_info,
                                     np.array(flat_solution),
                                     t_delay,
                                     show_plot=show_plot,
                                     show_node_features=show_node_features,
                                     show_legend=show_legend,
                                     save=save)
    else:
      print(f"No feasible solution found in seed {seed}")

    return t_travel, t_delay

# COMPUTING AND PRINTING THE SUMMARY OF STATISTICS
##################################################

for label, metrics in results.items():
  mean_tt = np.mean(metrics['travel_times'])
  std_tt = np.std(metrics['travel_times'])
  mean_out = np.mean(metrics['outs'])
  std_out = np.std(metrics['outs'])
  min_tt = np.min(metrics['travel_times'])
  max_tt = np.max(metrics['travel_times'])
  min_out = np.min(metrics['outs'])
  max_out = np.max(metrics['outs'])

  print(f"{label}")
  print(f"  Travel time: {mean_tt:.2f} ± {std_tt:.2f}")
  print(f"  Out-of-schedule: {mean_out:.2f} ± {std_out:.2f}")

# COMPARING PERFORMANCES OF THE MODELS USING A PLOT
##################################################

import matplotlib.pyplot as plt
import numpy as np

# Prepare data
labels = list(results.keys())# COMPARING PERFORMANCES OF THE MODELS USING A PLOT
##################################################

import matplotlib.pyplot as plt
import numpy as np

# Prepare data
labels = list(results.keys())
labels = labels[1:] # excluding no trained model

travel_means = [np.mean(results[key]['travel_times']) for key in labels]
travel_stds = [np.std(results[key]['travel_times']) for key in labels]
outs_means = [np.mean(results[key]['outs']) for key in labels]
outs_stds = [np.std(results[key]['outs']) for key in labels]

x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Performance Comparison", fontsize=16)

# --- Travel Time Plot ---
axs[0].bar(x, travel_means, yerr=travel_stds, capsize=5, color='skyblue')
axs[0].set_title("Total Travel Time")
axs[0].set_ylabel("Minutes")
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=45, ha='right')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# --- Out-of-schedule Deliveries Plot ---
axs[1].bar(x, outs_means, yerr=outs_stds, capsize=5, color='salmon')
axs[1].set_title("Out-of-Schedule Deliveries")
axs[1].set_ylabel("Accumulated time")
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels, rotation=45, ha='right')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust layout to make space for the title
plt.savefig("comparison_n50.png", dpi=300, bbox_inches='tight')
plt.show()