import os
import numpy as np
import pandas as pd
from functools import reduce
import sys


def parseField(input_path):
    
    """
    Get IDs and values of volume and metadata files
    
    Params:
        - input_path (str): path of TXT file to analyze
        
    Returns:
        - ids (np.array of ints): array with subject IDs
        - values (np.array of floats): array with read field
        
    """
    
    with open(input_path) as f: entries = f.readlines()
    entries.pop(0)

    ids = [f.split(",")[0].replace("\"", "") for f in entries]
    values = [f.split(",")[1].replace("\"", "").replace("\n", "") for f in entries]

    (ids, values) = zip(*sorted(zip(ids, values)))

    ids = np.array(ids).astype("int")
    values = np.array(values).astype("float")
    
    nan = np.where(np.isnan(values))
    
    if len(nan) > 0:
        ids = np.delete(ids, nan)
        values = np.delete(values, nan)
    
    return (ids, values)

def combineInfo(ids,values):
    """
    Combine information from several files into an only pairing of IDs and
    values. IDs and values will be sorted according to IDs values

    Parameters
    ----------
    ids : list of np.ndarray
        IDs to combine
    values : list of np.ndarray
        Corresponding values ot combine

    Returns
    -------
    combined_ids : np.ndarray
        Final combined IDs
    combined_values : np.ndarray
        Final combined values

    """
    
    cont = 0
    for i,value in zip(ids,values):
        if cont == 0:
            combined_ids = np.copy(i).astype(int)
            combined_values = np.copy(value).astype(float)
        else:
            combined_ids = np.concatenate((combined_ids, i.astype(int))).astype(int)
            combined_values = np.concatenate((combined_values, value.astype(float))).astype(float)
        cont += 1
     
    # Sort IDs and values
    sort = np.argsort(combined_ids)
    combined_ids = combined_ids[sort]
    combined_values = combined_values[sort]
    
    unique_ids = np.unique(combined_ids)

    
    if np.array_equal(unique_ids,combined_ids):
        # Non-overlapping sets
        return (combined_ids, combined_values)
    else:
        # Overlapping sets
        final_values = []
        for u_id in unique_ids:
            ind = np.where(combined_ids == u_id)[0]
            if len(ind) == 1: # Only 1 occurrence, copy value
                final_values.append(combined_values[ind])
            else: # Several occurrences of ID, take mean
                final_values.append(np.mean(combined_values[ind]))
                
        final_values = np.array(final_values).astype(float)
        
        return (unique_ids, final_values)
    
    
def loadData(file):
    """
    Load data from file or from a set of files with IDs and values

    Parameters
    ----------
    file : str or list of str
        Input file with IDs and values information

    Returns
    -------
    ids : np.ndarray 
        Extracted IDs
    values : np.ndarray
        Extracted values
    name : str
        Variable name

    """
    
    if file == "":
        print("Unspecified dependent variable file or independent variable file. Please provide valid files")
        sys.exit(2)
    
    # If string inputs are provided, adequate them into list
    if type(file) == str and "," in file:
        file = file.split(",")
    elif type(file) == str and not("," in file):
        file = [file]
        
    if len(file) > 1:
        file_ids = []
        file_values = []
        for n,f in enumerate(file):
            if not(os.path.exists(f)) or os.path.isdir(f):
                print("File {} does not exist or is a directory. Please provide a valid file".format(f))
                sys.exit(2)
            (file_id, file_value) = parseField(f)
            file_ids.append(file_id)
            file_values.append(file_value)
            if n == 0:
                filename = f.split("/")[-1]
                name = filename.split(".")[0]
        (file_ids, file_values) = combineInfo(file_ids, file_values)
        
    elif len(file) == 1:
        if not os.path.exists(file[0]) or os.path.isdir(file[0]):
            print("File {} does not exist or is a directory. Please provide a valid file".format(file))
            sys.exit(2)
        (file_ids, file_values) = parseField(file[0])
        filename = file[0].split("/")[-1]
        name = filename.split(".")[0]
    
    return (file_ids, file_values, name)



def buildDataframe(files,names):
    """
    Set data from X, Y, and covariate files in a common Pandas dataframe

    Params:
        - files : list
            Set of files to be read
        - names : list of str
            Set of variable names to be assigned

    Outputs:
        - df : Pandas dataframe
            Dataframe with all common information stored
True
    """    
    
    read_dfs = []
    
    for file,name in zip(files,names):
        (ids,values,_) = loadData(file)
        read_df = pd.DataFrame(values,columns=[name],index=ids)
        read_df.index.name = "eid"
        read_dfs.append(read_df)
        
    df = reduce(lambda  left,right: pd.merge(left,right,on=['eid'],
                                            how='inner'), read_dfs)
    
    return df


# Set an output folder for the parameter files
outfolder = "/media/name/data0/imiomics/parameters_files/" # Folder where to save parameter files

if not(os.path.exists(outfolder)):
    os.makedirs(outfolder)

f = "txt" # Type of file where to save parameter files for Imiomics ("txt" or "csv", default is "txt")
csv_name = None # Name for CSV file, in case we want to save the parameter files as CSV

# TXT files with reference IDs in the HDF5 files for Imiomics for females and males
txt_files = ["female_ids.txt","male_ids.txt"]
    
# Metadata files (always include sex.txt)
files = ["metadata/sex.txt",
         "metadata/age.txt"]

names = ["sex","age"] # Metadata names

# Obtain common dataframe
df = buildDataframe(files,names).dropna()

# Separate sex files
genders = ["female","male"]

for i,gender in enumerate(genders):
    gender_df = df[df["sex"]==i]
    ids = np.array(gender_df.index).astype(int)
    
    # Filter gender IDs with those IDs located in the HDF5 files
    ref_ids = np.loadtxt(txt_files[i]).astype(int)

    ids,ind,_ = np.intersect1d(ids,ref_ids,return_indices=True)
    gender_df = gender_df.iloc[ind]
    
    # Extract columns of each gender and save them in a txt or csv, according to the f parameter
    if f.lower() == "csv":
        gender_df["eid"] = ids.copy()
        if csv_name is None:
            gender_df.to_csv(os.path.join(outfolder,"{}_parameters.csv".format(gender)))
        else:
            gender_df.to_csv(os.path.join(outfolder,"{}_{}.csv".format(gender,csv_name)))
    else: # Write TXT files in any other case
       columns = gender_df.columns
       np.savetxt(os.path.join(outfolder,"{}_ids.txt".format(gender)),ids,fmt="%i")
       for c in columns:
           if c != "sex":
               array = np.array(gender_df[c]).astype(float)
               np.savetxt(os.path.join(outfolder,"{}_{}.txt".format(gender,c)),array,fmt="%f")
