import torch
state = torch.load('ml/checkpoints/latest.pt', map_location='cpu')
nans = [k for k,v in state.items() if torch.isnan(v).any() or torch.isinf(v).any()]
print('NaN tensors:', nans)
