import torch
import torchvision
from torch import nn
def create_effnetb2_model(num_classes:int = 101,
                          seed:int = 42,
                          ):
  effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  effnetb2_transforms = effnetb2_weights.transforms()
  model = torchvision.models.efficientnet_b2(weights = effnetb2_weights)
  for param in model.parameters():
    param.requires_grad = False


  torch.manual_seed(42)
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1408, out_features=num_classes, bias=True)
  )
  return model, effnetb2_transforms
