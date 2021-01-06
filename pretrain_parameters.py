import torch
import torchvision
import collections
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
pre_state_dict = load_state_dict_from_url(model_urls,progress=True)

dicts = collections.OrderedDict()

for k,value in pre_state_dict.items():
  if 'features' in k:
    k = k.replace('features', 'conv_layers')
  # if 'classifier' in k:
  #   k = k.replace('classifier', 'fcs')
  dicts[k] = value

torch.save(dicts, "./vgg16_bn.pth")
