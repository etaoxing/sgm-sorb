from pud.dependencies import *

def variance_initializer_(tensor, scale=1.0, mode='fan_in', distribution='uniform'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale /= max(1., fan_in)
    elif mode == "fan_out":
        scale /= max(1., fan_out)
    else:
        raise ValueError

    if distribution == 'uniform':
        limit = math.sqrt(3.0 * scale)
        nn.init.uniform_(tensor, -limit, limit)
    else:
        raise ValueError


def untorchify(tensor):
    return tensor.cpu().detach().numpy()


def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_env_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed) # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/