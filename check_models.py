import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load the .env file to get your API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        print("Fetching available models that support 'generateContent'...\n")
        
        # List all models
        for m in genai.list_models():
            # Check if 'generateContent' is a supported method for this model
            if 'generateContent' in m.supported_generation_methods:
                print(f"--- Found a usable model ---")
                print(f"Model name: {m.name}")
                print(f"Description: {m.description}\n")

    except Exception as e:
        print(f"An error occurred while trying to list models: {e}")