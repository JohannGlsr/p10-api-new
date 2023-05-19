from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule
import multiprocessing
from io import BytesIO
#app
app = Flask(__name__)

def create_architecture(archi_name, encoder_name, encoder_weights):
    if archi_name == "Unet":
        return  smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=n_classes)
    elif archi_name == "FPN":
        return  smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=n_classes)
    elif archi_name == "DeeplabV3":
        return  smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=n_classes)
    else:
        raise ValueError(f"Architecture invalide : {archi_name}")

# Importation de la classe LightningModule
class OurModel(LightningModule):
  
  # Méthode d'initialisation de la classe, qui prend en entrée l'architecture de l'encodeur et les poids de l'encodeur 
  def __init__(self, archi_name, encoder_name, encoder_weights):
    
    super(OurModel,self).__init__()

    self.layer = create_architecture(archi_name, encoder_name, encoder_weights)
    self.lr=1e-3 
    self.batch_size=2 
    self.numworker=multiprocessing.cpu_count()//2
    
  def forward(self,x):
    return self.layer(x)


nom_de_architecture = "Unet"
nom_de_l_encodeur = "resnet18"
poids_de_l_encodeur = "imagenet"
n_classes = 8

model = OurModel(archi_name=nom_de_architecture, encoder_name=nom_de_l_encodeur, encoder_weights=poids_de_l_encodeur)

model.load_state_dict(torch.load('Unet.resnet18.imagenet.pth'))

# Move the model to GPU
model = model.cuda()

colors = [
        [0, 0, 0], # Sans importances
        [128, 64, 128], # Flat
        [70, 70, 70], # Construction
        [153, 153, 153], # Object
        [107, 142, 32], # Nature
        [0, 129, 180], # Sky
        [219, 20, 59], # Human
        [0, 0, 142], # Vehicle
    ]

transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def segmentedImageToColor(temp):
    
    # Obtenir la hauteur et la largeur de l'image segmentée "temp"
    height = temp.shape[0]
    width = temp.shape[1]
    
    # Créer un tableau rempli de zéros, avec la hauteur et la largeur de "temp" et trois canaux de couleurs (rouge, vert et bleu)
    rgb = np.zeros((height, width, 3))
    
    # Parcourir les indices de couleurs et les valeurs de couleurs correspondantes
    for l, color in enumerate(colors):
        
        # Sélectionner les pixels de "temp" ayant la même couleur que l'indice en cours
        matching_pixels = temp == l
        
        # Affecter la valeur de couleur correspondante aux pixels sélectionnés dans le tableau rgb
        rgb[matching_pixels] = color
    
     # Normalisation des valeurs des canaux de couleurs de rgb entre 0 et 1
    rgb = rgb / 255.0
    
    return rgb

@app.route('/predict', methods=['POST'])
def flip_vertical():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    image = request.files['image']
    img = Image.open(image)

    # Prétraitement de l'image
    input_image = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(input_image)

    from torchvision import transforms
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    inv_img = inv_normalize(input_image.squeeze(0)).cpu()
    output_x = output.detach().cpu().squeeze(0)
    decoded_output = segmentedImageToColor(torch.argmax(output_x, 0))
    
    decoded_output_image = Image.fromarray((decoded_output * 255).astype(np.uint8))

    # Renvoi de l'image

    # Renvoi de l'image
    flipped_image_bytes = BytesIO()
    decoded_output_image.save(flipped_image_bytes, format='JPEG')  # Convertit l'image en bytes
    flipped_image_bytes.seek(0)  # Place le curseur au début des données
    return flipped_image_bytes, 200, {'Content-Type': 'image/jpeg'}

#if __name__ == '__main__':
#    app.run(debug=True)
