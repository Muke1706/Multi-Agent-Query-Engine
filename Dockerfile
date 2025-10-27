# 1. Start with an official Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
#    We use --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. (Optional) Expose a port if this were a web server (we'll need this later)
# EXPOSE 8000

# 7. Define the default command to run when the container starts
#    For now, it will just run our agent_graph script
CMD ["python", "agent_graph.py"]