import argparse
import torch
import json
from PIL import Image


from functions import load_checkpoint, process_image, predict

parser = argparse.ArgumentParser(description='arparse for making prediction with neural network')

# add argument for save dir
parser.add_argument('--save_dir', action='store', dest='save_dir', default='checkpoint.pth',
                   help='Enter the location that contains the trained model. Default is checkpoint.pth')

# add argument for the image path
parser.add_argument('--image_path', action='store', dest='image_path', default='flowers/test/12/image_04012.jpg',
                   help='Enter path to image')

# add argument for topk
parser.add_argument('--top_k', action='store', dest='topk', type=int, default=5,
                   help='Enter the top k')

# add argument to on gpu
parser.add_argument('--gpu', action='store', dest='gpu', default=False,
                   help='Enter the gpu mode true or false. The default mode is False')



inputs = parser.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
   
# load saved model
model_v1 = load_checkpoint(inputs.save_dir)
#print(model_v1.classifier)

# set an image path for test
path = inputs.image_path

# open origial image
image = Image.open(path)
probs, labels = predict(path, model_v1, inputs.topk, inputs.gpu)

names = []

for i in labels:
    names.append(cat_to_name[i])

# print the name of the flower with the highest probability.
print("Predicted flower name is {}".format(names[0]))


