from dataclasses import dataclass # Provides a decorator and functions for automatically adding generated special methods, such as __init__() and __repr__() to user defined classes.
import torch # Module with everything about PyTorch

def get_random_problems(b, n): # Generate tensors that define scenarios
  # b: batch_size
  # n: problem_size, it is the number of customers (without considering the depot)

  # Coordinate generation
  depot_xy = 12*torch.rand(size=(b, 1, 2)) # Generate a random tensor with values in (0, 12) km with the specified shape
  # shape: (b, 1, 2)
  customers_xy = 12*torch.rand(size=(b, n, 2))
  # shape: (b, n, 2)

  # Other feature generation
  customers_demand = torch.randint(1, 10, size=(b, n)) # Generate a random tensor with values in [1, 9] units with the specified shape
  customers_service = (3 + (5 - 3)*torch.rand(size = (b, n))) # Service times between (3, 5) min
  customers_deadline = (60 + (480 - 60)*torch.rand(size = (b, n))) # Deadlines between (60, 480)
  # shape: (b, n)
  random_multipliers = 1.0 + torch.rand(size = (b, n+1, n+1))
  # shape: (b, N, N)

  return depot_xy, customers_xy, customers_demand, customers_service, customers_deadline, random_multipliers # Return the four tensors (features for a whole batch)

@dataclass # With the decorator avoid defining constructor, the next variables will be initialized as init parameters
class Reset_State: # Class to restart the feature tensors
  depot_xy: torch.Tensor = None
  # shape: (b, 1, 2)
  customers_xy: torch.Tensor = None
  # shape: (b, n, 2)
  customers_demand: torch.Tensor = None
  customers_service: torch.Tensor = None
  customers_deadline: torch.Tensor = None
  # shape: (b, n)
  random_multipliers: torch.Tensor = None
  # shape: (b, N, N)

@dataclass
class Step_State: # Class to define the tensors that will be used to update the state of the env (they will be inputs of the NN-model and the Transition Function)
  t: int = None # The time-step (t) for decoding computations
  L_max: int = None # Max capacity of the vehicle
  
  a_t: torch.Tensor = None # Tensor of actions (a_t). It indicates which locations have been visited for all the samples and trajectories
  L_t: torch.Tensor = None # Follow up on the total number of supplies in the vehicle
  t_travel: torch.Tensor = None # Follow up on the transcurred time from the beggining to the end of the solution
  t_out: torch.Tensor = None # Follow up on the delays of the solution to be penalized
  xi_trajectory: torch.Tensor = None # Binary to know if the trajectory has finished
  BATCH_IDX: torch.Tensor = None # Binary to index other tensors
  POMO_IDX: torch.Tensor = None # Binary to index other tensors
  # shape: (b, pomo)
  xi_t: torch.Tensor = None # Mask for every location in the instance
  # shape: (b, pomo, N)
  H_cal: torch.tensor = None
  # shape: (b, pomo, 0~)

class CVRPEnv: # Main class that defines the environment (states, actions, transition and reward functions)
  def __init__(self, **env_params): # Env params are just problem_size and pomo_size
    # Const @INIT
    ####################################
    self.env_params = env_params # A dict with information to set up the environment

    # Set env_params as attributes of the class
    self.problem_size = env_params['problem_size'] # What we call n (without counting the depot)
    self.pomo_size = env_params['pomo_size'] # What we call Gamma

    # These elements do not change through an episode
    ####################################
    self.batch_size = None
    self.L_max = None # Total capacity of the vehicle
    
    self.depot_customers_xy = None # Coordinates for all the nodes
    # shape: (b, N, 2)
    self.depot_customers_demand = None # Demands for all the nodes (the depot should have demand 0)
    self.depot_customers_service = None # Service times for all the nodes (the depot should have a time-service of 15)
    self.depot_customers_deadline = None # Deadlines for all the nodes (the depot should have a long deadline)
    # shape: (b, N)
    self.BATCH_IDX = None
    self.POMO_IDX = None
    # shape: (b, pomo)
    self.random_multipliers = None

    # Dynamic elements of the state (the state itself)
    ####################################
    self.t = None # Time-step
    self.a_t = None # Actions
    self.L_t = None # Tensor to follow up on the total amount of supplies in the vehicle for every sample and trajectory
    self.d_travel = None # Tensor to follow up on the distance traveled for every sample and trajectory
    self.t_travel = None # Tensor to follow up on the transcurred time from the beggining to the end of the solutions for every sample and trajectory
    self.t_out = None # Tensor to follow up on the delays of the solutions for every sample and trajectory
    # shape: (b, pomo)
    self.H_cal = None # It store the solutions for every instance and trajectories
    # shape: (b, pomo, 0~)

    # Masks and flags
    ####################################
    self.xi_visit = None # Visit mask, it is independent from the demand mask
    self.xi_t = None # Visit and demand mask (the combination gotten from an OR operation)
    # shape: (b, pomo, n + 1)
    self.at_the_depot = None # Binary tensor to know if the visit location is the depot
    self.xi_trajectory = None # Binary to know if a solution has already been finished
    # shape: (b, pomo)

    # States to return (classes defined above)
    ####################################
    self.reset_state = Reset_State() # An instantiation of the class Reset_State, it puts all the elements of the state to None
    self.step_state = Step_State() # An instantiation of the class Step_State, it initialize all the elements of the state

  def load_problems(self, batch_size): # Set some of the class attributes (the static ones)
    self.batch_size = batch_size

    # Generate instances
    depot_xy, customers_xy, customers_demand, customers_service, customers_deadline, random_multipliers = get_random_problems(batch_size, self.problem_size) # Generate scenarios (batches of data)

    # Save the generated tensors as the corresponding attributes of the class
    depot_zero_feature = torch.zeros(size=(self.batch_size, 1)) # A utility tensor
    # shape: (b, 1)
    self.depot_customers_xy = torch.cat((depot_xy, customers_xy), dim=1) # Concatenate coordinates of the depot and deliveries
    # shape: (b, N, 2)
    self.depot_customers_demand = torch.cat((depot_zero_feature, customers_demand), dim=1) # Concatenate coordinates of the depot and deliveries
    self.depot_customers_service = torch.cat((depot_zero_feature, customers_service), dim=1) # Concatenate time services of the depot and deliveries
    self.depot_customers_deadline = torch.cat((depot_zero_feature, customers_deadline), dim=1) # Concatenate deadlines of the depot and deliveries
    # shape: (b, N)
    self.random_multipliers = random_multipliers

    # Set specific time-service for the depot
    self.depot_customers_service[:, 0] = 15 # Set service time of the depot as 1 (deterministic time to fill-up the capacity of the vehicle)
    self.depot_customers_deadline[:, 0] = float('inf') # Set deadline of the depot as a very long number

    # These tensors will be used to index the mask and the prob tensors
    self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
    self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    # Set the attributes of the reset_state object
    self.reset_state.depot_xy = depot_xy # Change from None to the depot_xy
    self.reset_state.customers_xy = customers_xy
    self.reset_state.customers_demand = customers_demand
    self.reset_state.customers_service = customers_service
    self.reset_state.customers_deadline = customers_deadline
    self.reset_state.random_multipliers = random_multipliers

    # Set the attributes of the step_state object
    self.step_state.BATCH_IDX = self.BATCH_IDX
    self.step_state.POMO_IDX = self.POMO_IDX

  def reset(self): # Setup the tensors and other variables that follow up on the state and the solution (the dynamic elements)
    self.t = 0 # Time-step (an int)
    self.a_t = None # Tensor of actions
    # shape: (b, pomo)
    self.H_cal = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long) # H_cal (the solutions for all the samples and trajectories)
    self.edge_times = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
    # shape: (b, pomo, 0~)

    # Assigning the max capacity of the vehicle based on the size of the instance
    if self.problem_size == 20:
      self.L_max = 30 # For 20 locations, 30 is the total capacity
    elif self.problem_size == 30:
      self.L_max = 35 # For 30 locations, 35 is the total capacity
    elif self.problem_size == 50:
      self.L_max = 40 # For 50 locations, 40 is the total capacity
    elif self.problem_size == 100:
      self.L_max = 50 # For 100 locations, 50 is the total capacity
    else:
      raise NotImplementedError

    self.L_t = torch.ones(size=(self.batch_size, self.pomo_size))*self.L_max # Available supplies in the vehicle
    self.d_travel = torch.zeros(size=(self.batch_size, self.pomo_size)) # Distance traveled for every sample and trajectory
    self.t_travel = torch.zeros(size=(self.batch_size, self.pomo_size)) # Transcurred time from the beginning to the end of the solution for every sample and trayectory
    self.t_out = torch.zeros(size=(self.batch_size, self.pomo_size)) # Delays for every sample and trajectory
    self.xi_trajectory = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
    self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool) # Boolean tensor to indicate if the vehicle is in the depot
    # shape: (b, pomo)

    self.xi_visit = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
    self.xi_t = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
    # shape: (b, pomo, n + 1)
  
    reward = None
    done = False

    return self.reset_state, reward, done

  def pre_step(self): # Setup the attributes of the step_state object defined in the constructor (this will serve as input to NN-model)
    self.step_state.t = self.t
    self.step_state.L_max = self.L_max
    self.step_state.a_t = self.a_t
    self.step_state.L_t = self.L_t
    self.step_state.t_travel = self.t_travel
    self.step_state.t_out = self.t_out
    self.step_state.xi_t = self.xi_t
    self.step_state.xi_trajectory = self.xi_trajectory
    self.step_state.H_cal = self.H_cal
    reward = None
    done = False

    return self.step_state, reward, done # To avoid the attributes of step_state in None there must be a statement where load_problems is called

  def step(self, a_current): # Execute transitions and compute reward function
    # a_current.shape: (batch, pomo), the tensor of actions

    # Update prev, current actions and solution 
    ####################################
    self.t += 1 # Increase time-step
    if self.t == 1:
      a_prev = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
    else:
      a_prev = self.a_t # Before updating current action, save the selected indexes in a previous step for dist and time computations
      # shape: (batch, pomo)
    self.a_t = a_current # Update current actions
    self.at_the_depot = (a_current == 0) # Binary indicating if the depot was visited
    # shape: (batch, pomo)
    self.H_cal = torch.cat((self.H_cal, self.a_t[:, :, None]), dim=2) # Update solutions
    # shape: (batch, pomo, 0~)

    # Setup tensors to follow up on the features associated with the solutions for all the samples and all the POMO trajectories
    ####################################
    coordinate_list = self.depot_customers_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, -1) # Set a tensor of coordinates to be available for every trajectory
    # shape: (batch, pomo, N, 2)
    demand_list = self.depot_customers_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1) # Set a tensor of demands to be available for every trajectory
    service_list = self.depot_customers_service[:, None, :].expand(self.batch_size, self.pomo_size, -1) # Set a tensor of service times to be available for every trajectory
    deadline_list = self.depot_customers_deadline[:, None, :].expand(self.batch_size, self.pomo_size, -1) # Set a tensor of deadlines to be available for every trajectory
    # shape: (batch, pomo, N)
    gathering_index = a_current[:, :, None] # Unsqueeze selected indexes
    # shape: (batch, pomo, 1)
    random_multipliers_list = self.random_multipliers[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, self.problem_size + 1) # Add the POMO size dimension
    # shape: (batch, pomo, N, N)

    # Filtering features associated with current selection
    ####################################
    selected_coordinates_prev = coordinate_list[self.BATCH_IDX, self.POMO_IDX, a_prev] # Filter the coordines of those locations visited in the previous step
    selected_coordinates = coordinate_list[self.BATCH_IDX, self.POMO_IDX, a_current] # Filter the coordines of those locations visited in the current step
    # shape: (batch, pomo, 2)
    selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # Filter the demands of those locations visited in the previous step
    selected_service = service_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # Filter the service times of those locations visited in the previous step
    selected_deadline = deadline_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # Filter the deadlines of those locations visited in the previous step
    selected_random_multipliers = random_multipliers_list[self.BATCH_IDX, self.POMO_IDX, a_prev, a_current] # Filter the random multipliers 
    # shape: (batch, pomo)

    # Update tensors that follow up on the solutions
    ####################################
    edge_dist = torch.sqrt((selected_coordinates[:, :, 0] - selected_coordinates_prev[:, :, 0] )**2 + (selected_coordinates[:, :, 1] - selected_coordinates_prev[:, :, 1] )**2)
    self.d_travel += edge_dist # Update travel distance
    edge_time = selected_random_multipliers*edge_dist # Computing edge time as a stochastic function of the edge distance
    self.t_travel += edge_time # Update travel time according to the time to travel the edge
    self.t_out += torch.maximum(self.t_travel - selected_deadline, torch.zeros_like(self.t_out)) # Update delays just after arriving the location
    self.L_t -= selected_demand # Remove the demand for the deliveries already made
    self.L_t[self.at_the_depot] = self.L_max # Refill supplies if the depot was visited
    self.t_travel += selected_service # Update travel time according to the time services

    # Update mask
    self.xi_visit[self.BATCH_IDX, self.POMO_IDX, a_current] = float('-inf') # Update visit mask
    self.xi_visit[:, :, 0][~self.at_the_depot] = 0  # Depot is considered unvisited, unless you are at the depot
    # shape: (batch, pomo, problem + 1)

    self.xi_t = self.xi_visit.clone() # Equals the two tensors considered as masks
    round_error_epsilon = 0.00001
    demand_too_large = self.L_t[:, :, None] + round_error_epsilon < demand_list # Which requests have a demand greater than the available amount of supplies
    self.xi_t[demand_too_large] = float('-inf') # Block those requests greater than load
    # shape: (batch, pomo, problem + 1)

    newly_finished = (self.xi_visit == float('-inf')).all(dim=2) # Evaluates if the condition is meet for every element in the tensor (get rid of the last dimension of visited...)
    self.xi_trajectory = self.xi_trajectory + newly_finished # Update finished mask using the above tensor, the + operation works as an OR
    # shape: (batch, pomo)

    # Do not mask depot for finished episodes (handling variable length solutions)
    self.xi_t[:, :, 0][self.xi_trajectory] = 0

    # update StepState instantiation (input to the NN-model and the step function)
    self.step_state.t = self.t
    self.step_state.L_t = self.L_t
    self.step_state.t_travel = self.t_travel
    self.step_state.t_out = self.t_out
    self.step_state.a_t = self.a_t
    self.step_state.xi_t = self.xi_t
    self.step_state.xi_trajectory = self.xi_trajectory
    self.step_state.H_cal = self.H_cal

    done = self.xi_trajectory.all() # finish until all the routes have been constructed
    if done:
      reward = -1*(self.t_travel + self.t_out)
    else:
      reward = None

    return self.step_state, reward, done # Return updated state, reward and done