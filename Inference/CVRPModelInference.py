import torch
import torch.nn as nn
import torch.nn.functional as F

class CVRPModel(nn.Module):
    def __init__(self, **model_params): # The model_params argument is a dict with hyperparameters for Neural Network generation
        super().__init__() # Seek for the attributes from the parent class (nn.Module)
        self.model_params = model_params # Save the model parameters dict as an attribute of the class

        # Instantiation of the encoder and decoder structures
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None # To save embeddings (node representations)
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state): # Set features for computation in their respective attributes

        # Load and scale features before feeding the model
        depot_xy = reset_state.depot_xy / 12 # In reset_state the features of the problem are storaged
        # shape: (batch, 1, 2)
        node_xy = reset_state.customers_xy / 12
        # shape: (batch, problem + 1, 2)
        customers_demand = reset_state.customers_demand / 9
        customers_service = reset_state.customers_service / 15
        customers_deadline = reset_state.customers_deadline / 480
        # shape: (batch, problem + 1)

        # Concatenate delivery node features
        node_features = torch.cat((node_xy, customers_demand[:, :, None], customers_service[:, :, None], customers_deadline[:, :, None]), dim=2) # Create a global feature tensor (without depot features)
        # shape: (batch, problem + 1, 5)

        self.encoded_nodes = self.encoder(depot_xy, node_features) # Compute encoding nodes using the encoder
        # shape: (batch, problem + 1, embedding)
        self.decoder.set_kv(self.encoded_nodes) # Method from the decoder, note that it is just an instruction (return anything), it serves to set up key and value tensors

    def forward(self, state): # State is an instantiation of the State_Step class from the CVRPEnv module
        batch_size = state.BATCH_IDX.size(0) # Save batch and Gamma sizes from indexing tensors
        pomo_size = state.BATCH_IDX.size(1)

        if state.t == 0:  # First move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long) # create a_1 tensor (the first tensor of actions for every sample and trajectory, all routes start at the depot)
            prob = torch.ones(size=(batch_size, pomo_size)) # The first decisions are deterministic, they will not affect the NN-parameters update

        elif state.t == 1:  # Second move, POMO
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size) # use all the locations as second actions, defining the possible trajectories
            prob = torch.ones(size=(batch_size, pomo_size)) # Second actions are also deterministic

        else: # Sampling indexes to construct solutions
            encoded_last_node = _get_encoding(self.encoded_nodes, state.a_t) # The function _get_encoding is external and is defined below to gather the vector representations associated with the current actions
            # shape: (batch, pomo, embedding)
            probs = self.decoder(state.L_max, encoded_last_node, state.L_t, state.t_travel, state.t_out, ninf_mask=state.xi_t) # Compute probs using decoder
            # shape: (batch, pomo, problem + 1)

            if self.training or self.model_params['eval_type'] == 'softmax': # Stochastic sampling (only during training), self.training is a attribute from the parent class torch.nn
                while True:  # To fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad(): # Disable gradient tracking for all the tensors in the next operations (this, in general, saves memory)
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size) # Flat the tensor (multinomial accepts matrix or vectors), select the index based on the probabilities, and reshape
                    # "\" indicates the compiler that the current line is interrupted and it continues in the next
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all(): # if none selected probability equals to cero then get out of the while loop (avoiding selecting 0 probability elements)
                        break

            else: # Greedy sampling (only during inference or tests instance)
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # Value not needed during inference. It can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick): # Gathers the vector representations associated with the current actions
    # encoded_nodes.shape: (batch, problem + 1, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # Layers for initial embeddings
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)]) # Stack of encoder layers (blocks)

    def forward(self, depot_xy, node_features):
        # depot_xy.shape: (batch, 1, 2)
        # node_features.shape: (batch, problem, 5)

        # Apply Linear Transforms to get embeddings
        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_features)
        # shape: (batch, problem, embedding)

        # Concatenate embeddings to get encoder layers input
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem + 1, embedding)

        # Get vector representations for every location with the stack of encoder layers
        for layer in self.layers: # Loop over the stack of encoder layers
            out = layer(out)

        return out # Return vector representations
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Layers for Multi-Head Attention (note last dimension is the product of head and qkv_dim)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Sub-layers for Transformers, they are loaded from external functions defined below (instance norm is used rather than batch norm)
        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1): # Transformer block
        # input1.shape: (batch, problem+1, embedding), node representations
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num) # Apply a function defined below. It literally performs a reshape
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v) # Don't mask anything!
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem+1, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # Key for multi-head attention
        self.v = None  # Value, for multi-head_attention
        self.single_head_key = None  # For single-head attention

    def set_kv(self, encoded_nodes): # This is the function used in the preforward method from the encoder
        # It computes and reshapes the k,v and single_head_key attributes
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, L_max, encoded_last_node, load, t_travel, t_out, ninf_mask): # The mask ninf_mask is provided independently
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # t_travel.shape: (batch, pomo)
        # t_out.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem + 1)

        # Scale context features (dynamic state of the problem)
        load = load / L_max
        t_travel_scaler = t_travel.max() if t_travel.max() > 0 else 1
        t_travel = t_travel / t_travel_scaler
        t_out_scaler = t_out.max() if t_out.max() > 0 else 1
        t_out = t_out / t_out_scaler

        #print(f"\tChecking Load: {load.shape}\n{load[0]}")
        #print(f"\tChecking t_travel: {t_travel.shape}\n{t_travel[0]}")
        #print(f"\tChecking t_out: {t_out.shape}\n{t_out[0]}")

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None], t_travel[:, :, None], t_out[:, :, None]), dim=2) # Use enc..., load and t_travel to compute the query (different from the AM)
        # shape = (batch, pomo, embedding + 3)

        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num) # setting up query
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask) # MHA for the decoder, this time a mask is used
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask # Mask again, before softmax (note that the mask is aggregated not indexed)

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem + 1)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num): # Reshape and permute dimensions (initial setting up to get qkv)
    # q.shape: (batch, n+1, head_num*key_dim)   : n+1 can be either 1 or problem_size

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n+1, key_dim): n+1 can be either 1 or problem_size
    # k, v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    # Apply masks (they are aggregated to the tensor not indexed)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled) # Attention tensor
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans

class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1))) # This sintax helps to optimize computation avoiding steps in the management of memory