import torch
state = torch.load("classifier.pth", map_location="cpu")
print([k for k in state.keys()][:20])
