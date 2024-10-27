import streamlit as st
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import pytesseract
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import PyPDF2
import psycopg2
from psycopg2 import Binary
from datetime import datetime
import os
import io
import traceback
import logging
from googleapiclient.discovery import build
import google.generativeai as genai
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(filename='app_log.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# YouTube API setup
YOUTUBE_API_KEY = "AIzaSyBhqC7XAtWuZdZSPjPFfzkIgmE_UBVqyOk"  # Replace with your actual YouTube API key
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Gemini API setup
GOOGLE_API_KEY = "AIzaSyAkQmZm6ayyy0AajZEh1FN7Ms5IrdbOVbQ"  # Replace with your actual Google API key
genai.configure(api_key=GOOGLE_API_KEY)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return clip_model, clip_processor, blip_processor, blip_model
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        st.error("Failed to load necessary models. Please check the logs for more information.")
        return None, None, None, None

clip_model, clip_processor, blip_processor, blip_model = load_models()

# Load the segmentation model
class YourSegmentationModel(nn.Module):
    def __init__(self):
        super(YourSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # For grayscale, change input channels to 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)  # Output 1 channel for binary segmentation

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv_final(x)  # Final convolution to get segmentation map
        return x

# Update the path to segformer_finetuned.pth
segformer_path = os.path.join(base_dir, 'segformer_finetuned.pth')

# Load the segmentation model weights
try:
    loaded_weights = torch.load(segformer_path)
    segmentation_model = YourSegmentationModel()
    segmentation_model.load_state_dict(loaded_weights, strict=False)
    segmentation_model.eval()
except FileNotFoundError:
    logging.error(f"Segmentation model file not found: {segformer_path}")
    st.error(f"Error: Could not find the file {segformer_path}")
    st.error("Please make sure the segformer_finetuned.pth file is in the same directory as this script.")
except Exception as e:
    logging.error(f"Error loading segmentation model: {str(e)}")
    st.error("Failed to load the segmentation model. Please check the logs for more information.")

# Load the ADR data
adr_data_path = r"C:\Users\Tanusree\NEURODEGENERATION\New folder\frontendwork\drugs_side_effects_drugs_com.csv"
adr_data = pd.read_csv(adr_data_path)

# Make sure the column names match what's used in the function
if 'Medicine' not in adr_data.columns or 'ADR' not in adr_data.columns:
    adr_data = adr_data.rename(columns={'Drug': 'Medicine', 'Side Effects': 'ADR'})

# Medicine identification
def identify_medicine(image):
    medicines = adr_data['Medicine'].dropna().astype(str).tolist()
    inputs = clip_processor(text=medicines, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds
    similarity = torch.cosine_similarity(image_embeddings, text_embeddings)
    best_match_index = similarity.argmax().item()
    identified_medicine = medicines[best_match_index]
    
    # Retrieve medicine use and side effects
    adr_info = adr_data[adr_data['Medicine'] == identified_medicine]
    if not adr_info.empty:
        side_effects = adr_info['ADR'].values[0]  # Get side effects
    else:
        side_effects = "No information available."
    
    return identified_medicine, side_effects

# Preprocessing for grayscale MRI image segmentation
def preprocess_image_for_segmentation(image):
    if image.mode != 'L':  # Check if the image is grayscale
        image = image.convert('L')  # Convert the image to grayscale mode
    
    target_size = (224, 224)
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Postprocessing for segmentation with sharpening and dilation
def postprocess_segmentation(segmentation_output):
    if segmentation_output.dim() == 4:  # Expecting [batch_size, 1, height, width]
        segmented_image = segmentation_output.squeeze().cpu().numpy()
        segmented_image = (segmented_image - segmented_image.min()) / (segmented_image.max() - segmented_image.min() + 1e-8)  # Normalize output
        binary_mask = (segmented_image > 0.5).astype(np.uint8)  # Convert to binary mask
        kernel = np.ones((3, 3), np.uint8)  # Define kernel size
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)  # Dilation
        return Image.fromarray((dilated_mask * 255).astype('uint8'))
    else:
        raise ValueError(f"Unexpected shape for segmentation output: {segmentation_output.shape}")

# Segmentation function
def segment_image(image):
    image_tensor = preprocess_image_for_segmentation(image)
    with torch.no_grad():
        segmentation_output = segmentation_model(image_tensor)
    segmented_image = postprocess_segmentation(segmentation_output)
    return segmented_image  # Return the processed image

# Function to extract text from uploaded files
def extract_text_from_file(file):
    try:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
        else:
            image = Image.open(io.BytesIO(file.read()))
            return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"Error extracting text from file: {str(e)}")
        st.error("Failed to extract text from the uploaded file. Please check the logs for more information.")
        return ""

# PostgreSQL connection
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname="neurodegenerative",
            user="postgres",
            password="5291tanusree",
            host="localhost",
            port=5432
        )
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        st.error("Failed to connect to the database. Please check the logs for more information.")
        return None

# Create tables
def create_tables():
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS patient_details (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    disease TEXT,
                    medicine TEXT,
                    food_intake TEXT,
                    mri_image BYTEA,
                    segmented_image BYTEA,
                    report_text TEXT,
                    medicine_image BYTEA,
                    date_added TIMESTAMP
                )
            ''')
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error creating tables: {str(e)}")
            st.error("Failed to create necessary database tables. Please check the logs for more information.")

create_tables()

# Function to get YouTube videos
def get_youtube_videos(query, max_results=5):
    try:
        search_response = youtube.search().list(
            q=query,
            type='video',
            part='id,snippet',
            maxResults=max_results
        ).execute()

        videos = []
        for search_result in search_response.get('items', []):
            video = {
                'title': search_result['snippet']['title'],
                'description': search_result['snippet']['description'],
                'url': f"https://www.youtube.com/watch?v={search_result['id']['videoId']}"
            }
            videos.append(video)
        
        return videos
    except Exception as e:
        logging.error(f"YouTube API error: {str(e)}")
        return []

# Function to get video description based on user query
def get_video_description(image, user_query):
    try:
        # First, get a basic description of the image
        inputs = blip_processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        base_description = blip_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Return the base description along with the user query
        return f"Image Description: {base_description}\n\nUser Query: {user_query}"
    except Exception as e:
        logging.error(f"Error in video description: {str(e)}")
        return "Error generating video description"

# Function to analyze food intake using Gemini
def analyze_food_intake(disease, food_intake):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Analyze the following food intake for a patient with {disease}. Provide a brief analysis of the nutritional value and potential risks or benefits for the patient's condition:\n\n{food_intake}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error analyzing food intake: {str(e)}")
        return "Error analyzing food intake"

# Function to generate a graph of food intake
def generate_food_intake_graph(food_intake):
    food_items = food_intake.split(',')
    food_counts = {}
    for item in food_items:
        item = item.strip()
        if item in food_counts:
            food_counts[item] += 1
        else:
            food_counts[item] = 1
    
    plt.figure(figsize=(10, 5))
    plt.bar(food_counts.keys(), food_counts.values())
    plt.title("Food Intake Frequency")
    plt.xlabel("Food Items")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt

# Helper function to store data in database
def store_in_database(data, medicine_image):
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            current_time = datetime.now()
            
            cur.execute('''
                INSERT INTO patient_details 
                (name, age, gender, disease, medicine, food_intake, report_text, medicine_image, date_added)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                data['name'], data['age'], data['gender'], data['disease'], data['medicine'],
                data['food_intake'], data['report_text'], 
                Binary(medicine_image.getvalue()) if medicine_image else None,
                current_time
            ))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error storing data in database: {str(e)}")
            st.error("Failed to store data in the database. Please check the logs for more information.")

def image_segmentation_page():
    st.header("Image Segmentation")
    mri_image = st.file_uploader("Upload MRI/CT Image", type=["jpg", "png", "jpeg"])
    if mri_image:
        mri_img = Image.open(mri_image)
        segmented_image = segment_image(mri_img)
        st.image(segmented_image, caption="Segmented Image")
        
        # Save segmented image to session state for use in other pages
        st.session_state['segmented_image'] = segmented_image

def form_filling_page():
    st.header("Patient Information")
    with st.form("patient_form"):
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Patient Age", min_value=0, max_value=150, step=1)
        patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])
        disease_discovered = st.text_input("Disease Discovered")
        medicine_intake = st.text_input("Medicine Intake (comma-separated)")
        food_intake = st.text_area("Food Intake for Each Day (comma-separated)")
        
        report_file = st.file_uploader("Upload Medical Report", type=["pdf", "jpg", "png", "jpeg"])
        medicine_image = st.file_uploader("Upload Medicine Image", type=["jpg", "png", "jpeg"])

        submitted = st.form_submit_button("Submit")

    if submitted:
        if all([patient_name, patient_age, patient_gender, disease_discovered, medicine_intake, food_intake, report_file, medicine_image]):
            # Process and store data
            report_text = extract_text_from_file(report_file)
            med_img = Image.open(medicine_image)
            identified_medicine, side_effects = identify_medicine(med_img)
            
            # Store in session state for use in visualization page
            st.session_state['patient_data'] = {
                'name': patient_name,
                'age': patient_age,
                'gender': patient_gender,
                'disease': disease_discovered,
                'medicine': medicine_intake,
                'food_intake': food_intake,
                'report_text': report_text,
                'identified_medicine': identified_medicine,
                'side_effects': side_effects
            }
            
            # Store in database
            store_in_database(st.session_state['patient_data'], medicine_image)
            
            st.success("Data submitted successfully. Please go to the Visualization page to see the results.")
        else:
            st.error("Please fill out all required fields and upload all required files.")

def visualization_page():
    st.header("Results Visualization")
    
    if 'patient_data' not in st.session_state:
        st.warning("Please submit patient information first.")
        return
    
    data = st.session_state['patient_data']
    
    # Display patient information
    st.subheader("Patient Information")
    st.write(f"Name: {data['name']}")
    st.write(f"Age: {data['age']}")
    st.write(f"Gender: {data['gender']}")
    st.write(f"Disease: {data['disease']}")
    
    # Display segmented image if available
    if 'segmented_image' in st.session_state:
        st.image(st.session_state['segmented_image'], caption="Segmented MRI/CT Image")
    
    # Analyze food intake
    food_analysis = analyze_food_intake(data['disease'], data['food_intake'])
    st.subheader("Food Intake Analysis")
    st.write(food_analysis)
    
    # Generate and display food intake graph
    food_graph = generate_food_intake_graph(data['food_intake'])
    st.pyplot(food_graph)
    
    # Display medicine information
    st.subheader("Medicine Information")
    st.write(f"Prescribed Medicine: {data['medicine']}")
    st.write(f"Identified Medicine: {data['identified_medicine']}")
    st.write(f"Potential Side Effects: {data['side_effects']}")
    
    # Get relevant YouTube videos
    st.subheader("Relevant Videos")
    videos = get_youtube_videos(f"{data['disease']} {data['identified_medicine']} medical information")
    for video in videos[:3]:  # Display top 3 videos
        st.video(video['url'])

def main():
    st.title("AI-Powered Healthcare System")
    
    pages = {
        "Image Segmentation": image_segmentation_page,
        "Patient Information": form_filling_page,
        "Results Visualization": visualization_page
    }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    pages[selection]()

if __name__ == "__main__":
    main()
