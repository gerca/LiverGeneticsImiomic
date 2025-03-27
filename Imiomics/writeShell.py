import numpy as np
import os

# Write Shell files for multiple linear or logistic regression Imiomics computations in Bianca

# IDs used in the Imiomics computations
info_male_file = "male_ids.txt"
info_female_file = "female_ids.txt"


# Number of male and female IDs used
n_men = len(np.loadtxt(info_male_file).astype(int))
n_women = len(np.loadtxt(info_female_file).astype(int))

N = [str(n_men), str(n_women)]

time = 40
# Parameters to read the HDF5 files and name the related files to the Imiomics pipeline

types = ["Jac","Fat"]
genders = ["M","F"]
full_genders = ["male","female"]
hdf5_types = ["_jac_medianNew", "_ff_medianNew"] # Tags of HDF5 files to be read
groups = ["volume","fat"]

fat = "" # Leave this parameter like this for now
split = info_female_file.split("/")[-1]
split = split.split("_")[-1]
#variable = split.split(".")[0]
variable = "VAT" # Main variable to be computed 

hdf5_folder = "/proj/sens2019016/UKB_ImageData/NewRegistrations_medianNew/HDF5/" # HDF5 folder to be read

masks = "_bodymask_medianNew.vtk" # Bodymasks tags to be read

compute = "_Multiple"
covariates = ["A"] # Shortened name for the covariates used
covariates_full = ["age","asat","TLT","VAT"] # Name of covariates
covariates_join = "_".join(covariates)

parameter_folder = "/proj/sens2019016/Software/Py_Pipeline/Name/ParameterFiles/" # Parameter folder with parameter files that have to be read

out = "/imiomics/ShellFiles/{}_{}{}/".format(variable,covariates_join,compute)

if not os.path.exists(out):
    os.makedirs(out)
    
if out[-1] != "/":
    out += "/"

for i,typ in enumerate(types):
    cont = 0
    for n,gender in zip(N,genders):
        
        elements = [compute, typ, fat, gender, variable, covariates_join,n]
        elements_join = "_".join(elements)
        file = compute + "_" + typ + "_" + fat + "_" + gender + "_" + variable + "_" + covariates_join
        
        covariate_section = [parameter_folder + full_genders[cont] + "_" + covariate + ".txt" for covariate in covariates_full]
        covariate_section = ",".join(covariate_section)
        
        with open(out + file + ".sh", "w") as f:
            f.write("#!/bin/bash -l\n")
            f.write('\n')
            f.write('#SBATCH -p node -n 1 \n')
            f.write('#SBATCH -A sens2019016 \n')
            f.write('#SBATCH -t ' + str(time) + ':00:00 \n');
            f.write('#SBATCH -J ' + file + ' \n')
            f.write('\n')
            f.write('module load python3/3.9.5\n')
            f.write('python3 /proj/sens2019016/Software/Py_Pipeline/Name/Codes/regression.py -f ' + hdf5_folder + full_genders[cont] + hdf5_types[i] + ".hdf5 -g " + groups[i] + " -v " + covariate_section + " -i " + parameter_folder + full_genders[cont] + "_ids.txt -l " + parameter_folder + full_genders[cont] + "_" + variable + ".txt -m /proj/sens2019016/Software/Py_Pipeline/Imiomics_masks/" + full_genders[cont] + masks + " -c " + compute[1:].lower() + " -o /proj/sens2019016/Software/Py_Pipeline/ResultFiles/" + elements_join + ".nrrd")
            f.close()
   
        cont += 1
        
        
bianca_folder = os.path.join("/proj/sens2019016/Software/Py_Pipeline/Name/ShellFiles",variable) # Shell files folder in Bianca
        
sh_files = sorted(os.listdir(out))

with open(os.path.join(out,"overall.sh"),"w") as f:
    
    f.write("#!/bin/bash -l\n\n")  

    for sh_file in sh_files: 
        
        f.write("sbatch {}\n".format(sh_file))
