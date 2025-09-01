import pandas as pd
from astartes.molecules import train_test_split_molecules #Import 

def create_train_test_scaffold(df, smiles, group, test_size):
    '''
    Create train test scaffold.
    -----
    Parameters:
    df: DataFrame
    smiles: smiles column name
    group: toxic or non-toxic
    test_size: Ratio of test size
    ----
    Return x_train, x_test, y_train, y_test
    '''
    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split_molecules(molecules=df[smiles], y=df[group], test_size=float(test_size),
    train_size=float(1.0-test_size), sampler="scaffold", random_state=0)
    #Dataframe
    x_train = pd.DataFrame(x_train, y_train.index, columns=[smiles])
    x_test  = pd.DataFrame(x_test,  y_test.index, columns=[smiles])
    y_train = pd.DataFrame(y_train, y_train.index, columns=[group])
    y_test  = pd.DataFrame(y_test, y_test.index, columns=[group])
    return x_train, x_test, y_train, y_test

def main():
    print("This software is starting to split your data using random split")
    #df = pd.read_csv("pubchem_no_duplicates.csv", index_col = "cid")
    smiles = "canonical_smiles"
    group = "Toxicity Value"
    
    test_size = 0.3
    train_size = float(1-test_size)
    print("Your test set is ", test_size, " and your train set is ", train_size)
    df = pd.read_csv('capsule_x_train.csv')
    print(df[smiles])
    print(df[group])
    
    x_train, x_test, y_train, y_test = create_train_test_scaffold(df, smiles, group, test_size)
    print("No of compounds in x_train = ", len(x_train), "No of compounds in y_train ", len(y_train))
    print("No of compounds in x_test = ",  len(x_test), "No of compounds in y_test ", len(y_test))

    #save
    x_train.to_csv("capsule_x_train_subset.csv")
    x_test.to_csv("capsule_x_test_subset.csv")
    y_train.to_csv("capsule_y_train_subset.csv")
    y_test.to_csv("capsule_y_test_subset.csv")

    print("#"*100)
    print('Finished!')

if __name__ == "__main__":
    main()
