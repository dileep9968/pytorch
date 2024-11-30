import gradio as gr
import os
import torch

from model import create_effnetb2_model
from typing import Tuple, Dict
from timeit import default_timer as timmer
from typing import Tuple, Dict

class_names = ['pizza','steak','sushi']

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3)

# loead save weights
effnetb2.load_state_dict(torch.load(
    f='foodvision_mini/09_pretrained_effnetb2_features_extractor_pizza_steak_20_percent.pth',
    map_location = torch.device('cpu')
                         )
      )

def predict(img) -> Tuple[Dict, float]:

  # start timer 
  start_time = timmer()
  # transform the input image
  img = effnetb2_transforms(img).unsqueeze(0)  # unsqeeze the add the batch dimesnion

  # put model into eval mode
  effnetb2.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnetb2(img), dim=1)

  #create prediction 
  pred_labels_and_prob = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # end timer and calculate time
  end_time = timmer()
  pred_time = round(end_time-start_time,4)

  return pred_labels_and_prob, pred_time

title = 'FoodVision Mini'

description = "EffnetB2"
article = 'Pytorch model deployment'

example_list = [['examples/'+example] for example in os.listdir('examples')]

# create the gradio demo

demo = gr.Interface(fn = predict,
                    inputs = gr.Image(type='pil'),
                    outputs = [gr.Label(num_top_classes=3,label='Predictions'),
                               gr.Number(label="Prediction time (s)")],
                    examples = example_list,
                    title = title,
                    description = description,
                    article = article)
demo.launch(debug=False, share=True)
