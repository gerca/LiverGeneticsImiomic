#!/usr/bin/env Rscript


# uwemenzel@gmail.com 



## +++ Convert file format


## +++ Call:

# convert_phenofile file1 file2 file3 ... outfile





## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)     

if(length(args) < 2) {
  cat("\n")
  cat("  Insufficient number of arguments\n")
  cat("  Usage: convert_phenofile  <infile1>  <infile2>  <infile3> ... <outfile> \n") 
  cat("  All infiles must exist, outfile will be created.\n") 
  cat("\n")
  quit("no")
}



## +++ Infiles & outfile names 

infiles = list() # list with infiles
for (i in 1:(length(args)-1)) infiles[i] = args[i]

cat("\n")
for (i in 1:length(infiles)) cat(paste("  Infile", i, ":", infiles[i], "\n"))    
cat("\n")

outfile = args[length(args)]
cat(paste("  Outfile :", outfile, "\n\n"))

if(file.exists(outfile)) {
  cat(paste("\n  File", outfile, "already exists.\n")) 
  cat("  Delete the existing file or choose another outfile name.\n\n")
  quit("no")
} 





## +++ Load infiles:

phenofiles = list()  
for (i in 1:length(infiles)) {
  file = as.character(infiles[i])
  if(!file.exists(file)) { 
    stop(paste("\n\n  ERROR (convert_phenofile.R): File '", file, "' not found. \n\n")) 
  } else {
    phenofiles[[i]] = read.csv(file)  
  }
}






## +++ One infile only:

if(length(phenofiles) == 1) {

  outframe = phenofiles[[1]]
  rownames(outframe)  = outframe[,1]   # expect sample names in 1st column
  outframe[,1] <- NULL

  id_cols = data.frame(FID = row.names(outframe), IID = row.names(outframe)) 
  rownames(id_cols) = rownames(outframe)

  outframe = merge(id_cols, outframe, by = "row.names") 
  outframe$Row.names <- NULL
  colnames(outframe)[1] = "#FID"    # requested by plink

  write.table(outframe, file = outfile, row.names = FALSE, quote = FALSE, sep = "\t")    
  cat(paste("\n  Output file", outfile, "written.\n\n"))

  quit("no")

} 






## +++ Check if all phenofiles have the same sample names

#   if all phenofiles have exactly the same sample names, we can merge them to a single output file.
#   otherwise , multiple output files are created. 

samples = list()
for (i in 1:length(phenofiles)) samples[[i]] = phenofiles[[i]][,1]
 
pairs = combn(1:length(samples),2)  
ok = 0
for (i in 1:ncol(pairs)) { 
  i1 = pairs[1,i]
  i2 = pairs[2,i]
  cat(paste("  Comparing sample names:", i1, "with", i2, ": "))
  c1 = length(samples[[i1]]) == length(samples[[i2]])
  c2 = length(intersect(samples[[i1]], samples[[i2]])) == length(samples[[i2]])
  if (c1 & c2) {
    cat("same sample names\n") 
    ok = ok +1 
  } else { 
    cat("different sample names\n")
  }
}
cat("\n")

if(ok != ncol(pairs)) {
  cat("  You don't have identical sample names in all input files.\n")
  cat("  Please, use only files with identical sample names as input (or run single files only)\n\n")
  quit("no")
}






## +++ Merge frames if same sample names  

outframe = phenofiles[[1]]
rownames(outframe)  = outframe[,1]   # expect sample names in 1st column
outframe[,1] <- NULL

for (i in 2:ncol(pairs)) { 
  nextframe = phenofiles[[i]]
  rownames(nextframe)  = nextframe[,1]  # expect sample names in 1st column 
  nextframe[,1] <- NULL 
  outframe = merge(outframe, nextframe, by = "row.names")  
  rownames(outframe) = outframe$Row.names
  outframe$Row.names <- NULL
}




## +++ Convert to format requested by plink 

id_cols = data.frame(FID = row.names(outframe), IID = row.names(outframe)) 
rownames(id_cols) = rownames(outframe)

outframe = merge(id_cols, outframe, by = "row.names") 
outframe$Row.names <- NULL
colnames(outframe)[1] = "#FID"    # requested by plink




 
## +++ Save outframe  

write.table(outframe, file = outfile, row.names = FALSE, quote = FALSE, sep = "\t")    
cat(paste("\n  Output file", outfile, "written.\n\n"))


 






