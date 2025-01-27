#!/usr/bin/env Rscript


# uwemenzel@gmail.com 




## +++ Get a random sample of participants
#     parents: filter_samples.R 





## +++ Call:

## PGEN:    
# random_samples ukb_imp_v3_chr19.psam  10000  sampled.txt  

# BED:
# random_samples  FTD_chr22.fam  50000 s50000.txt  

# RAW, 2 columns:
# random_samples  filter_input.txt  1000  s1000.txt

# RAW, 1 column:
# random_samples  filter_in_one.txt  10000  choose10000.txt



 

## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)    

if(length(args) != 3) {
  cat("\n")
  cat("  Usage: random_samples   <infile>   <number>  <outfile> \n\n") 
  cat("         <infile>:  Input file with sample ID's (.psam, .fam or raw text file with 1 or 2 columns)\n") 
  cat("         <number>:  Number of random participants to fetch \n")
  cat("         <outfile>: Output file name \n\n") 
  quit("no")
}

infile = args[1]    
number = as.integer(args[2]) 
outfile = args[3]  





## +++ Infiles & outfiles  

if(!file.exists(infile)) {
  cat(paste("\n  ERROR (random_samples.R): File", infile, "not found.\n\n")) 
  quit("no")
} 


if(file.exists(outfile)) {
  cat(paste("\n  File", outfile, "already exists.\n")) 
  cat("  Delete the existing file or choose another outfile name.\n\n")
  quit("no")
} 





## +++ Load infile  (to sample from)   (Detect input file format)

line1 = read.table(infile, header = FALSE, sep = "\t", nrow = 1)

if(ncol(line1) == 3) {    
  cat("\n  Input file is probably a .psam file.\n\n")
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)  
  input_samples = indata[,2]  
  if(class(input_samples) != "integer") {
    cat(paste("\n  ERROR (random_samples.R): The 2nd column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(input_samples)
    cat("\n\n")
    quit("no")
  } 
}

if(ncol(line1) == 6) {   
  cat("\n  Input file is probably a .fam file.\n\n")
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)  
  input_samples = indata[,2]  
  if(class(input_samples) != "integer") {
    cat(paste("\n  ERROR (random_samples.R): The 2nd column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(input_samples)
    cat("\n\n")
    quit("no")
  } 
}

if(ncol(line1) == 2) {    
  cat("\n  Input file is probably a raw file with 2 columns.\n\n")
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE) 
  input_samples = indata[,2]  # V2
  if(class(input_samples) != "integer") {
    cat(paste("\n  ERROR (random_samples.R): The 2nd column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(input_samples)
    cat("\n\n")
    quit("no")
  } 
}

if(ncol(line1) == 1) {     
  cat("\n  Input file is probably a raw file with 1 column.\n\n")
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)  
  input_samples = indata[,1]  # V1
  if(class(input_samples) != "integer") {
    cat(paste("\n  ERROR (random_samples.R): The 1st column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(input_samples)
    cat("\n\n")
    quit("no")
  } 
}

  
cat(paste("  The input list contains", length(input_samples), "participants.\n\n"))
 
if(number >= length(input_samples)) {
  cat("  The number of samples to fetch is bigger than the number of samples in the input file.\n\n")  
  quit("no")
} 




## +++ Take random sample and save 

rsample = sample(input_samples, number)   # defaults to replace = FALSE
outframe = data.frame(FID = rsample, IID = rsample)  
write.table(outframe, file = outfile, col.names = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)


cat(paste("  Output saved to:", outfile, "\n\n")) 
cat("  Use the program 'extract_samples' to obtain a genotype dataset based on the obtained samples.\n\n")
quit("no")  # uwemenzel@gmail.com   









