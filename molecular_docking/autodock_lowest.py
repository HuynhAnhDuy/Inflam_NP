import os
import re
import pandas as pd

def read_ligand(ligand_id,txt_path):
    
    # Initialize a dictionary to store affinities for each ligand
    affinities = {ligand_id: [] for ligand_id in ligand_id}

    # Define a pattern to extract affinity values
    affinity_pattern = re.compile(r'^\s*\d+\s+([-]?\d+\.\d+)', re.MULTILINE)

    # Process each .txt file in the directory
    for file_name in os.listdir(txt_path):
        if file_name.endswith('.txt'):
            # Extract ligand ID from the file name
            ligand_id = os.path.splitext(file_name)[0]  # Removes '.txt' from the file name

            if ligand_id in affinities:
                file_path = os.path.join(txt_path, file_name)

                with open(file_path, 'r') as file:
                    content = file.read()

                    # Extract affinity values
                    matches = affinity_pattern.findall(content)
                    for match in matches:
                        affinity = float(match)
                        affinities[ligand_id].append(affinity)
    return affinities

def find_lowest(ligand_id,affinities,lowest_csv):
    # Find the lowest binding affinity for each ligand
    lowest_affinities = []
    for ligand in ligand_id:
        if affinities[ligand]:
            lowest_affinity = min(affinities[ligand])
            lowest_affinities.append({'LigandID': ligand, 'lowest_binding_affinity': lowest_affinity})
        else:
            # If no affinities found, handle as needed (e.g., add 'No Data')
            lowest_affinities.append({'LigandID': ligand, 'lowest_binding_affinity': 'No Data'})
    
    # Create a DataFrame and save to CSV
    df_lowest_affinities = pd.DataFrame(lowest_affinities)
    df_lowest_affinities.to_csv(lowest_csv, index=False)
    
    return print(f'Lowest binding affinities have been saved to {lowest_csv}')


def main():
    ligand_csv = os.path.join('output', 'train1.csv')
    # Read ligand IDs from the CSV file
    ligand_id = pd.read_csv(ligand_csv)['LigandID'].dropna().tolist()
    
    # Directory paths
    txt_path = 'output/txt/test'
    lowest_csv = os.path.join('output/txt/test', 'output_test_affinity.csv')
    affinities=read_ligand(ligand_id,txt_path)
    find_lowest(ligand_id,affinities,lowest_csv)

if '__name__'=='__main__':
        main()
    







