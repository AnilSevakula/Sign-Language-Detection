import os
import string

# Get all uppercase letters A-Z
labels=['Hello','Thank you','Welcome','Eat','Drink','Sorry','Please','Help',' I Love You','Yes',
'No','How are you','What are you doing','Good morning','Goodbye','How','What','Who','Where','Mother','Father','Brother','Sister','Danger','Sleep','Play','Bathroom']
# folders = list(string.ascii_uppercase)

# Create each folder in the current directory
for folder in labels:
    try:
        os.makedirs(folder)
        print(f"Folder '{folder}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder}' already exists.")
