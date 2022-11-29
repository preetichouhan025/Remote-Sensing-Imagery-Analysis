import torchvision.transforms as transforms
import torch
import PIL.Image as Image 

def predict_image_class(model, img_path, classes):

  img_transforms = transforms.Compose([transforms.Resize([64,64]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
                                     ])
  model = model.eval()
  image = Image.open(img_path)
  image = img_transforms(image).float()
  image = image.unsqueeze(0)

  output = model(image)
  _, predicted = torch.max(output.data,1)

  return classes[predicted.item()]