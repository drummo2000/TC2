import os
import joblib
import csv
import sys
import pandas as pd
sys.path.append('../Client/')
from DeepLearning.GetObservation import getObservationFull
from DeepLearning.GetActionMask import getActionMask
from Agents.AgentRandom2 import AgentRandom2
import numpy as np

directory = "StateActions2"
files = os.listdir(directory)
# Get the full path and creation time of each file, then sort by creation time
sorted_files = sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))

sorted_files_snippet = sorted_files[0:100]




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
with open('ReplicateUCT/CSV/state_mask_action.csv', 'r') as source_file, open('ReplicateUCT/CSV/state_mask_action_non_setup.csv', 'w', newline='') as target_file:
    reader = csv.DictReader(source_file)
    writer = csv.DictWriter(target_file, fieldnames=reader.fieldnames)

    writer.writeheader()  # Write the column headers to the output file

    # Iterate through each row in the source CSV
    for row in reader:
        # Check if the 9th last column equals '1'
        if row['State1729'] != '1' and row['State1730'] != '1':
            writer.writerow(row)  # Write the row to the new CSV file if condition is met

##########################################################################################################################################################

# Get action mask for each state

# dummy_agent = AgentRandom2("P-1", -1)

# csv_file_path = "ReplicateUCT/action_masks.csv"

# # Process each .joblib file and write to a CSV
# with open(csv_file_path, mode='w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     state_rows = [f'Action{i}' for i in range(486)]
#     writer.writerow([*state_rows])  # Column headers; modify as per your feature extraction logic

#     # Iterate through each file in the directory
#     for filename in sorted_files:
#         # Load the joblib file
#         file_path = os.path.join(directory, filename)
#         game_state, action = joblib.load(file_path)


#         # Get action mask from game_state
#         possible_actions = dummy_agent.GetPossibleActions(game_state, player=game_state.players[0])
#         action_mask, _ = getActionMask(possible_actions)
        
#         row = action_mask
#         # Write the extracted features and action to the CSV
#         writer.writerow(row)

##########################################################################################################################################################

# Add action mask csv to full

# # Load the two CSV files
# state_actions = pd.read_csv('ReplicateUCT/state_actions.csv')
# masks = pd.read_csv('ReplicateUCT/action_masks.csv')

# # Assuming csv1 and csv2 have the same order of rows and same number of rows
# combined_csv = pd.concat([state_actions, masks], axis=1)  # `axis=1` for column-wise concatenation

# # Save the combined CSV
# combined_csv.to_csv('ReplicateUCT/state_action_mask.csv', index=False)

##########################################################################################################################################################

# move action column to end

# # Load the CSV file
# df = pd.read_csv('ReplicateUCT/state_action_mask.csv')

# # Move a column to the end
# col_to_move = df.pop('Action')  # Replace 'column_name' with the name of your column
# df['Action'] = col_to_move  # Append it back to the DataFrame

# # Save the modified DataFrame back to a CSV
# df.to_csv('state_mask_action.csv', index=False)