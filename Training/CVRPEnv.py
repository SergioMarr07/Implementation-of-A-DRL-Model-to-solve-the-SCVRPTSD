from dataclasses import dataclass # Provides a decorator and functions for automatically adding generated special methods, such as __init__() and __repr__() to user defined classes.
import torch # Module with everything about PyTorch

def get_random_problems(b, n): # Generate tensors that define scenarios
  # b: batch_size
  # n: problem_size, it is the number of customers (without considering the depot)

  # Coordinate generation
  ###################################################################################################
  depot_xy = 12*torch.rand(size=(b, 1, 2)) # Generate a random tensor with values in (0, 12) (we're using km) with the specified shape
  customers_xy = 12*torch.rand(size=(b, n, 2))

  # Other feature generation
  ###################################################################################################
  customers_demand = torch.randint(1, 10, size=(b, n)) # Generate a random tensor with values in [1, 9] units with the specified shape
  customers_service = (3 + (5 - 3)*torch.rand(size = (b, n))) # Service times between (3, 5) min
  customers_deadline = (60 + (480 - 60)*torch.rand(size = (b, n))) # Deadlines between (60, 480)

  return depot_xy, customers_xy, customers_demand, customers_service, customers_deadline # Return the four tensors (features for a whole batch)

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

@dataclass
class Step_State: # Class to define the dynamic State of the Environment (it'll be the input to the NN-model and the Transition Function)
  t: int = None # The time-step for decoding computations
  L_max: int = None # Max capacity of the vehicle

  a_t: torch.Tensor = None # Tensor of actions. It indicates which locations have been visited for all the samples and trajectories at time-step t
  L_t: torch.Tensor = None # Follow up on the total number of supplies in the vehicle
  t_travel: torch.Tensor = None # Follow up on the transcurred time from the beggining to the end of the solution
  t_out: torch.Tensor = None # Follow up on the delays of the solution to be penalized
  xi_trajectory: torch.Tensor = None # Binary to know if the trajectory has finished
  BATCH_IDX: torch.Tensor = None # Tensor of ints to index other tensors
  POMO_IDX: torch.Tensor = None # Tensor of ints to index other tensors
  # shape: (b, pomo)

  xi_t: torch.Tensor = None # Mask for every location in the instance
  # shape: (b, pomo, n + 1)

  H_cal: torch.tensor = None # Solutions for all the samples and trajectories in the batch
  # shape: (b, pomo, 0~)

class CVRPEnv: # Main class that defines the environment (states, actions, the transition and reward function)
  def __init__(self, **env_params): # Env params are just problem_size and pomo_size

    self.env_params = env_params # A dict with information to set up the environment
    self.problem_size = env_params['problem_size'] # What we call n (without counting the depot)
    self.pomo_size = env_params['pomo_size'] # What we call Gamma

    # Static elements of the state
    ###################################################################################################
    self.batch_size = None # Number of samples to process in one shot
    self.L_max = None # Total capacity of the vehicle

    self.depot_customers_xy = None # Coordinates for all the nodes (included the depot)
    # shape: (b, N, 2)

    self.depot_customers_demand = None # Demands for all the nodes (the depot should have demand 0)
    self.depot_customers_service = None # Service times for all the nodes (the depot should have a time-service of 15)
    self.depot_customers_deadline = None # Deadlines for all the nodes (the depot should have a long deadline)
    # shape: (b, N)

    self.BATCH_IDX = None
    self.POMO_IDX = None
    # shape: (b, pomo)

    # Dynamic elements of the state
    ###################################################################################################
    self.t = None # Time-step
    self.a_t = None # Actions
    self.L_t = None # Tensor to follow up on the total amount of supplies in the vehicle for every sample and trajectory
    self.d_travel = None # Tensor to follow up on the distance traveled for every sample and trajectory
    self.t_travel = None # Tensor to follow up on the transcurred time from the beggining to the end of the solutions for every sample and trajectory
    self.t_out = None # Tensor to follow up on the delays of the solutions for every sample and trajectory
    # shape: (b, pomo)

    self.H_cal = None # Store the solutions for every sample and computed trajectory, it is not necessary during training
    # shape: (b, pomo, 0~)

    self.xi_visit = None # Visit mask
    self.xi_t = None # Visit and demand mask (this is obtained through an OR operation)
    # shape: (b, pomo, n + 1)

    self.at_the_depot = None # Binary tensor to know if the visit location is the depot
    self.xi_trajectory = None # Binary to know if a solution has already been finished
    # shape: (b, pomo)

    # Instantiation of the classes defined above
    ###################################################################################################
    self.reset_state = Reset_State() # An instantiation of the class Reset_State, it puts all the elements of the state to None
    self.step_state = Step_State() # An instantiation of the class Step_State, it initialize all the elements of the state

  def load_problems(self, batch_size): # Setting up the static class attributes 
    self.batch_size = batch_size

    # Generating synthetic instances
    ###################################################################################################
    depot_xy, customers_xy, customers_demand, customers_service, customers_deadline = get_random_problems(batch_size, self.problem_size) # Generate scenarios (batches of data)

    depot_zero_feature = torch.zeros(size=(self.batch_size, 1)) # A utility tensor
    # shape: (b, 1)

    # Concatenating features of the depot and customers and setting them as the corresponding attributes of the class
    ###################################################################################################
    self.depot_customers_xy = torch.cat((depot_xy, customers_xy), dim=1) 
    # shape: (b, N, 2), where N = n + 1

    self.depot_customers_demand = torch.cat((depot_zero_feature, customers_demand), dim=1)
    self.depot_customers_service = torch.cat((depot_zero_feature, customers_service), dim=1)
    self.depot_customers_deadline = torch.cat((depot_zero_feature, customers_deadline), dim=1)
    # shape: (b, N)

    # Setting some special depot features 
    self.depot_customers_service[:, 0] = 15 # The deterministic time to fill-up the capacity of the vehicle
    self.depot_customers_deadline[:, 0] = float('inf') # Deadline of the depot as a very long number

    # These tensors will be used to index the mask and the prob tensors
    self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
    self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    # Setting the attributes of the reset_state object
    self.reset_state.depot_xy = depot_xy # Change from None to the depot_xy
    self.reset_state.customers_xy = customers_xy
    self.reset_state.customers_demand = customers_demand
    self.reset_state.customers_service = customers_service
    self.reset_state.customers_deadline = customers_deadline

    # Setting the attributes of the step_state object
    self.step_state.BATCH_IDX = self.BATCH_IDX
    self.step_state.POMO_IDX = self.POMO_IDX

  def reset(self): # Setting up the dynamic class attributes
    self.t = 0 # Time-step

    self.a_t = None # Tensor of actions
    # shape: (b, pomo)

    self.H_cal = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long) # The solutions for all the samples and trajectories
    # shape: (b, pomo, 0~)

    # Determining the max capacity of the vehicle according to the problem size n
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
    self.xi_trajectory = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool) # Mask for completed solutions
    self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool) # Binary tensor to indicate if the vehicle is in the depot
    # shape: (b, pomo)

    self.xi_visit = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1)) # Visit mask
    self.xi_t = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1)) # Global mask
    # shape: (b, pomo, n + 1)

    reward = None
    done = False

    return self.reset_state, reward, done

  def pre_step(self): # Setting up the attributes of the step_state object defined in the constructor (the input to NN-model)
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

  def step(self, a_current): # The transition and reward functions are icnluded here
    # Update prev, current actions, and solution
    ###################################################################################################
    self.t += 1 # Increase time-step
    if self.t == 1: # In the first time step the vechicles must start at the depot
      a_prev = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
    else:
      a_prev = self.a_t # Before updating current actions, save the selected indexes in a previous step for dist and time computations
      # shape: (batch, pomo)
    self.a_t = a_current # Update current actions using the input of the method
    self.at_the_depot = (a_current == 0) # Update the binary tensor indicating if the depot has been visited
    # shape: (batch, pomo)

    self.H_cal = torch.cat((self.H_cal, self.a_t[:, :, None]), dim=2) # Update solutions
    # shape: (batch, pomo, 0~)

    # Setting up tensors to decoding properly
    ###################################################################################################
    coordinate_list = self.depot_customers_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, -1) # Set a tensor of coordinates to be available for every trajectory
    # shape: (batch, pomo, N, 2)

    demand_list = self.depot_customers_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1) # Set a tensor of demands to be available for every trajectory
    service_list = self.depot_customers_service[:, None, :].expand(self.batch_size, self.pomo_size, -1) # Set a tensor of service times to be available for every trajectory
    deadline_list = self.depot_customers_deadline[:, None, :].expand(self.batch_size, self.pomo_size, -1) # Set a tensor of deadlines to be available for every trajectory
    # shape: (batch, pomo, N)

    gathering_index = a_current[:, :, None] # Unsqueeze selected indexes
    # shape: (batch, pomo, 1)

    # Filtering features associated with the current selection
    ###################################################################################################
    selected_coordinates_prev = coordinate_list[self.BATCH_IDX, self.POMO_IDX, a_prev] # Filter the coordinates of those locations visited in the previous step
    selected_coordinates = coordinate_list[self.BATCH_IDX, self.POMO_IDX, a_current] # Filter the coordinates of those locations visited in the current step
    # shape: (batch, pomo, 2)

    selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # Filter the demands of those locations visited in the previous step
    selected_service = service_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # Filter the service times of those locations visited in the previous step
    selected_deadline = deadline_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # Filter the deadlines of those locations visited in the previous step
    # shape: (batch, pomo)

    # Update dynamic elements of the state
    ####################################
    edge_dist = torch.sqrt((selected_coordinates[:, :, 0] - selected_coordinates_prev[:, :, 0] )**2 + (selected_coordinates[:, :, 1] - selected_coordinates_prev[:, :, 1] )**2) # Computing Euclidian edge distance
    self.d_travel += edge_dist # Updating travel distance
    omega = 1 + torch.rand(1,).item() # Generating a random number between 1 and 2
    edge_time = omega*edge_dist # Computing edge time as a stochastic function of the edge distance
    self.t_travel += edge_time # Updating travel time according to the time to travel the edge
    self.t_out += torch.maximum(self.t_travel - selected_deadline, torch.zeros_like(self.t_out)) # Updating delays just after arriving the location
    self.L_t -= selected_demand # Removing the demand of the already made deliveries 
    self.L_t[self.at_the_depot] = self.L_max # Refilling supplies if the depot was visited
    self.t_travel += selected_service # Updating travel time according to the time services

    # Updating masks
    ###################################################################################################
    self.xi_visit[self.BATCH_IDX, self.POMO_IDX, a_current] = float('-inf') # Updating visit mask
    self.xi_visit[:, :, 0][~self.at_the_depot] = 0  # Depot is considered unvisited, unless you are at the depot
    # shape: (batch, pomo, N)

    self.xi_t = self.xi_visit.clone() # Equaling global mask with the information of the visit mask
    round_error_epsilon = 0.00001 # To avoid errors for numerical computation
    xi_demand = self.L_t[:, :, None] + round_error_epsilon < demand_list # Determing which requests have a demand greater than the available amount of supplies (the demand mask)
    self.xi_t[xi_demand] = float('-inf') # Blocking those requests greater than the load
    # shape: (batch, pomo, N)

    newly_finished = (self.xi_visit == float('-inf')).all(dim=2) # Evaluates if the condition is meet for every element in the tensor (get rid of the last dimension of visited...)
    self.xi_trajectory = self.xi_trajectory + newly_finished # Update finished mask using the above tensor, the + operation works as an OR
    # shape: (batch, pomo)

    # Do not mask depot for finished episodes (handling variable length solutions)
    self.xi_t[:, :, 0][self.xi_trajectory] = 0

    # Updating StepState instantiation
    ###################################################################################################
    self.step_state.t = self.t
    self.step_state.L_t = self.L_t
    self.step_state.t_travel = self.t_travel
    self.step_state.t_out = self.t_out
    self.step_state.a_t = self.a_t
    self.step_state.xi_t = self.xi_t
    self.step_state.xi_trajectory = self.xi_trajectory
    self.step_state.H_cal = self.H_cal

    # Computing reward
    ###################################################################################################
    done = self.xi_trajectory.all() # finish until all the trajectories have been constructed
    if done:
      reward = -1*(self.t_travel + self.t_out)
    else:
      reward = None

    return self.step_state, reward, done