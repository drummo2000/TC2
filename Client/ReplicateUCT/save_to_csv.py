import os
import joblib
import csv
from Game.CatanGame import GameState
from DeepLearning.GetObservation import getObservationFull
import numpy as np

# directory = "StateActions2"
# files = os.listdir(directory)
# # Get the full path and creation time of each file, then sort by creation time
# sorted_files = sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))

# sorted_files_snippet = sorted_files[0:100]

# csv_file_path = "state_actions.csv"




# # Process each .joblib file and write to a CSV
# with open(csv_file_path, mode='w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     state_rows = [f'State{i}' for i in range(1737)]
#     writer.writerow([*state_rows, 'Action'])  # Column headers; modify as per your feature extraction logic

#     # Iterate through each file in the directory
#     for filename in sorted_files:
#         # Load the joblib file
#         file_path = os.path.join(directory, filename)
#         game_state, action = joblib.load(file_path)


#         # Get state representation from GameState Object
#         features = getObservationFull(game_state)
#         row = np.append(features, action)

#         # Write the extracted features and action to the CSV
#         writer.writerow(row)

##########################################################################################################################################################

# Go through csv and and if the 8th last column is = 1, then save that column to a new csv

# Open the source CSV file and the target file where you'll save the filtered rows
with open('state_actions.csv', 'r') as source_file, open('state_action_setup.csv', 'w', newline='') as target_file:
    reader = csv.reader(source_file)
    writer = csv.writer(target_file)

    # Iterate through each row in the source CSV
    for row in reader:
        # Check if the 9th last column equals '1'
        if row[-8] == '1' or row[-9] == '1':
            writer.writerow(row)  # Write the row to the new CSV file if condition is met
