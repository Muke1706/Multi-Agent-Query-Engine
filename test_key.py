import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the .env file (should pick up your NEW key)
load_dotenv()

print("Attempting to use the API key...")

try:
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest")

    # Make one single, simple call
    response = llm.invoke("Tell me a very short joke.")

    print("\n--- SUCCESS ---")
    print(f"API Key seems to work. Response: {response.content}")

except Exception as e:
    print("\n--- FAILED ---")
    print(f"An error occurred: {e}")
    print("\nDouble-check:")
    print("1. Did you paste the NEW API key correctly into the .env file?")
    print("2. Did you save the .env file?")
    # Optionally print the key to verify, but be careful sharing it
    # loaded_key = os.getenv("GOOGLE_API_KEY")
    # print(f"3. Key loaded starts with: {loaded_key[:5]}... ends with: {loaded_key[-4:]}")