import torch

print(torch.__version__) 
print(torch.version.cuda)          # Should print something like '12.2'
print(torch.cuda.is_available())   # Should print True
print(torch.cuda.get_device_name(0))  # Your GPU name
