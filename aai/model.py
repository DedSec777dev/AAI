import kagglehub

# Download latest version
path = kagglehub.dataset_download("dhairya903/flights-in-india")

print("Path to dataset files:", path)