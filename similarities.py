
from models.model import load_model
from utils.data import get_num_classes
from utils.graphs import cosine_similarity_matrix, plot_top_similar_pairs
from pathlib import Path
import torch
import pandas as pd



project_dir = Path(__file__).resolve().parent
weights_path = project_dir / 'models' / 'checkpoints' / 'best_resnet50.pth'
data_dir     = project_dir / 'data' 
images_dir     = data_dir   / 'images'
loaders_dir  = project_dir / 'data'   / 'dataloaders' / 'dataloaders.pth'
similarity_plot_dir = project_dir / 'results' / 'plots' / 'similarity_plots'
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = 'data/images/n02088466-bloodhound/n02088466_1262.jpg'
num_classes = get_num_classes(images_dir)

# Check if the similarity plot directory exists, if not create it
similarity_plot_dir.mkdir(parents=True, exist_ok=True)


# Model and target layer
model = load_model(num_classes=num_classes, feature_extract=False)
checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
target_layer = model.layer4[-1]
model.to(device)
model.eval()


# Extract classifier weights
classifier_weights = model.fc.weight.data.clone()

sim_matrix = cosine_similarity_matrix(classifier_weights, device)

plot_top_similar_pairs(sim_matrix, top_n=20, show = False, saving_dir=similarity_plot_dir)
df_sim = pd.DataFrame(sim_matrix.cpu().numpy())
df_sim.to_csv( str(similarity_plot_dir) +"/cosine_similarity_matrix.csv", index=False)