from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule
import multiprocessing
from io import BytesIO
from collections import Counter

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from tqdm import tqdm

from metaseg.generator.automatic_mask_generator import SamAutomaticMaskGenerator
from metaseg.generator.build_sam import sam_model_registry
from metaseg.utils import (download_model, load_image, save_image, show_image,)

app = Flask(__name__)

#################################################
######## Définition de notre modèle Unet ######## 
#################################################

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
model = model.cpu()

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

def test_model(model, test_image_path, transform):
    model = model.cuda()
    model.eval()

    # Chargement de l'image à tester
    #image = Image.open(test_image_path)

    # Prétraitement de l'image
    input_image = transform(test_image_path).unsqueeze(0).cuda()

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
    
    return decoded_output

#################################################
######## Configuration du modèle MetaSeg ######## 
#################################################

class SegAutoMaskPredictor:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)

        return self.model

    def image_predict(
        self,
        source,
        model_type,
        points_per_side,
        points_per_batch,
        min_area,
        output_path="output.png",
        show=False,
        save=False,
    ):
        read_image = load_image(source)
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model, points_per_side=points_per_side, points_per_batch=points_per_batch, min_mask_region_area=min_area
        )

        masks = mask_generator.generate(read_image)

        sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        mask_image = np.zeros((masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3), dtype=np.uint8)
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        for i, ann in enumerate(sorted_anns):
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            color = colors[i % 256]
            for i in range(3):
                img[:, :, 0] = color[0]
                img[:, :, 1] = color[1]
                img[:, :, 2] = color[2]
            img = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))
            img = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
            mask_image = cv2.add(mask_image, img)

        combined_mask = mask_image
        # combined_mask = cv2.add(read_image, mask_image)
        
        self.combined_mask = combined_mask
        if show:
            show_image(combined_mask)

        if save:
            save_image(output_path=output_path, output_image=combined_mask)

        return masks

#################################################
######## Définition des classes des maks ######## 
#################################################

def replace_identical_pixels(image_segmented_metaseg_path, image_segmenté_model):
    
    # Charger l'image segmentée
    image_segmented_metaseg = Image.open(image_segmented_metaseg_path)

    # Convertir les images en tableaux numpy
    image_segmented_metaseg_array = np.array(image_segmented_metaseg)
    image_segmenté_model_array = np.array(image_segmenté_model)

    # Conversion au bon format
    image_segmenté_model_array = (image_segmenté_model_array * 255).astype(np.uint8)

    # Reshape du tableau pour avoir une liste d'images
    liste1 = image_segmented_metaseg_array.reshape(-1, 3)
    liste2 = image_segmenté_model_array.reshape(-1, 3)
    liste3 = liste1.copy()

    # Trouver les lignes identiques dans la première liste
    unique_rows = np.unique(liste1, axis=0)

    # Créer un dictionnaire pour stocker les indices de chaque ligne dans la première liste
    index_dict = {tuple(row): [] for row in unique_rows}

    # Parcourir la première liste et enregistrer les indices de chaque ligne
    for i, row in enumerate(liste1):
        index_dict[tuple(row)].append(i)

    # Remplacer les valeurs identiques dans la première liste par la ligne la plus présente dans la deuxième liste
    for row in unique_rows:
        indices = index_dict[tuple(row)]
        counter = Counter(map(tuple, liste2[indices]))
        most_common_row = counter.most_common(1)[0][0]
        for i in indices:
            liste3[i] = most_common_row

    liste3 = liste3.reshape(256, 512, 3)
    return liste3

#################################################
##### Modification de l'avant de la voiture ##### 
#################################################

image_correction_voiture_clean_path = 'image_correction_voiture_clean.png'

def copy_black_pixels(image_correction_voiture_clean_path, liste3):
    # Charger l'image2_color
    image2_color = np.array(Image.open(image_correction_voiture_clean_path))

    # Conversion au bon format
    image2_color = (image2_color * 255).astype(np.uint8)

    # Conversion des images en tableau
    liste3_array = np.array(liste3)
    image2_color_array = np.array(image2_color)

    # Copier les pixels noirs de l'image2_color_array et les coller sur liste3_array
    hauteur, largeur, _ = image2_color_array.shape
    zone_a_copier = image2_color_array[hauteur-100:hauteur, :, :]  # Les 100 dernières lignes de l'image2_color_array

    # Trouver les indices des pixels noirs dans la zone à copier
    pixels_noirs_indices = np.where(np.all(zone_a_copier == [0, 0, 0], axis=-1))

    # Copier les pixels noirs sur liste3_array
    for y, x in zip(pixels_noirs_indices[0], pixels_noirs_indices[1]):
        liste3_array[y+hauteur-100, x] = image2_color_array[y+hauteur-100, x]

    return liste3_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    image = request.files['image']
    input_image = Image.open(image)

    #################################################
    ###### Segmentation par le model "metaseg" ###### 
    ################################################

    results = SegAutoMaskPredictor().image_predict(
        source=input_image,
        model_type="vit_h",
        points_per_side=64, 
        points_per_batch=64,
        min_area=0,
        output_path="temp.png",
        show=False,
        save=True,
    )
    
    #################################################
    ####### Segmentation par le modèle "Unet" ####### 
    #################################################

    # Segmentation par le modèle "Unet"
    image_segmenté_model = test_model(model, input_image, transform)

    image_segmented_metaseg_path = 'temp.png'
    liste3 = replace_identical_pixels(image_segmented_metaseg_path, image_segmenté_model)

    #################################################
    ###### Correction de l'avant de la voiture ###### 
    #################################################

    decoded_output = copy_black_pixels(image_correction_voiture_clean_path, liste3)

    decoded_output_image = Image.fromarray(decoded_output)

    # Renvoi de l'image
    output_image_bytes = BytesIO()
    decoded_output_image.save(output_image_bytes, format='PNG')  # Convertit l'image en bytes
    output_image_bytes.seek(0)  # Place le curseur au début des données
    return output_image_bytes, 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run()
