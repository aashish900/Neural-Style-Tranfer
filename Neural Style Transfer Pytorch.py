
# coding: utf-8

# In[177]:


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import copy


# In[113]:


imsize = 512

loader = transforms.Compose([
            transforms.Resize(imsize),  # Scale imported image 
            transforms.ToTensor()])     # Transform it to a tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # Fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image

style_img = image_loader('C:/ProgramData/Anaconda3/share/jupyter/lab/Images/picasso.jpg')
content_img = image_loader('C:/ProgramData/Anaconda3/share/jupyter/lab/Images/dancing.jpg')

assert style_img.size() == content_img.size(),     "we need  to import style and content images of the same size"


# In[114]:


unloader = transforms.ToPILImage() # Reconvert into PIL image
plt.ion()

def imshow(tensor, title = None):
    image = tensor.cpu().clone() # Clone the tensor to not do changes on it
    image = image.squeeze(0) # Remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
        plt.pause(0.01) # Pause a bit so that plots are updated

plt.figure()
imshow(style_img, title = 'Style Image')

plt.figure()
imshow(content_img, title = 'Content Image')


# In[115]:


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()  # Super class to inherit all there is in ContentLoss class
        # We 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# In[116]:


def gram_matrix(input):
    a, b, c, d = input.size() 
    # a = batch_size(=1)
    # b = number of feature maps
    # (c, d) = dimensions of f. map(N = c*d)
    
    features = input.view(a * b, c * d) # Resize feature length
    
    G = torch.mm(features, features.t())
    
    # We 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


# In[117]:


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# In[118]:


cnn = models.vgg19(pretrained=True).features


# In[119]:


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# Create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # View the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Normalize img
        return (img - self.mean) / self.std


# In[120]:


# Desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # Normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # Just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []

    # Assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # Add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# In[121]:


input_img = content_img.clone()
# if you want to use a white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# Add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')


# In[122]:


def get_input_optimizer(input_img):
    # This line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# In[123]:


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps = 450,
                       style_weight = 1000000, content_weight = 1):
    
    """Run the style transfer."""
    print('Building the style transfer model..')
    
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # A last correction...
    input_img.data.clamp_(0, 1)

    return input_img


# In[124]:


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title = 'Output Image')

# Sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()


# In[179]:


torchvision.utils.save_image(output, 'C:/Users/Aashish/Desktop/Styleized_Image.png')

