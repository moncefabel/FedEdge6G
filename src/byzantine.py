import copy
import torch


def make_byzantine_flip(local_model, global_model, scale=1.0):
    """
    Gradient flip : inverse la direction de la mise à jour.
    
    w_byz = w_global - scale * (w_local - w_global)
          = (1 + scale) * w_global - scale * w_local
    """
    byzantine = copy.deepcopy(global_model)
    global_state = global_model.state_dict()
    local_state  = local_model.state_dict()

    byz_state = {}
    for key in global_state:
        delta        = local_state[key].float() - global_state[key].float()
        byz_state[key] = global_state[key].float() - scale * delta

    byzantine.load_state_dict(byz_state)
    return byzantine