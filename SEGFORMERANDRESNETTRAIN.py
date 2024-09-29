import os
import zipfile
import logging
import streamlit as st
import torch
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification, SegformerForSemanticSegmentation
from transformers import AdamW
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Custom dataset class
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}  # Map folder names to labels

        for label, folder in enumerate(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.label_map[folder] = label
                for img_file in os.listdir(folder_path):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(folder_path, img_file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to extract zip files
def extract_zip(zip_file, extraction_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
    logging.info(f"Extracted {zip_file} to {extraction_dir}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),
])

# Fine-tuning function for ResNet
def fine_tune_resnet(train_loader):
    logging.info("Starting training process for ResNet model...")
    model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50', num_labels=3, ignore_mismatched_sizes=True)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(10):  # Adjust epochs as needed
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training ResNet"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model_save_path = '/content/models/resnet_finetuned.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    logging.info(f"ResNet model saved at {model_save_path}")
    return running_loss / len(train_loader)

# Fine-tuning function for SegFormer
def fine_tune_segformer(train_loader):
    logging.info("Starting training process for SegFormer model...")
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0', num_labels=3, ignore_mismatched_sizes=True)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(10):  # Adjust epochs as needed
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training SegFormer"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model_save_path = '/content/models/segformer_finetuned.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    logging.info(f"SegFormer model saved at {model_save_path}")
    return running_loss / len(train_loader)

# Streamlit UI
st.title("Train SegFormer and ResNet Models")

zip_file_path = st.file_uploader(r"c:\Users\Tanusree\Desktop\NEURODEGENERATION\neuroniiimages.zip", type=['zip'])

if st.button("Start Training"):
    if zip_file_path is not None:
        extraction_dir = "extracted_files"
        os.makedirs(extraction_dir, exist_ok=True)
        with open(zip_file_path.name, "wb") as f:
            f.write(zip_file_path.getbuffer())
        
        extract_zip(zip_file_path.name, extraction_dir)

        # Prepare dataset
        dataset = CustomImageDataset(root_dir=extraction_dir, transform=transform)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Fine-tune ResNet
        with st.spinner("Training ResNet..."):
            final_loss_resnet = fine_tune_resnet(train_loader)
        st.success(f"ResNet Training Complete with Final Loss: {final_loss_resnet}")

        # Fine-tune SegFormer
        with st.spinner("Training SegFormer..."):
            final_loss_segformer = fine_tune_segformer(train_loader)
        st.success(f"SegFormer Training Complete with Final Loss: {final_loss_segformer}")
    else:
        st.error("Please upload a valid ZIP file.")
