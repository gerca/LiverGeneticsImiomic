#!/usr/bin/env Rscript


# uwemenzel@gmail.com 




## +++ Fetch a phenotype by UKBB field ID:
#     (possibly use search_filedID first)  




## +++ Call:
#
# called by fetch_pheno.sh 
#
# fetch_pheno.R  ${phenofile}  ${headerfile}  ${field}  ${outfile}




## +++ Libraries, functions

library(data.table) # fread 




## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)     

if(length(args) < 4) {
  cat("\n")
  cat("  Usage: fetch_pheno.R  <phenofile>   <headerfile>   <fieldID>   <outfile>\n") 
  cat("      uwe.menzel@medsci.uu.se\n") 
  cat("\n")
  quit("no")
}

phenofile = args[1]	
headerfile = args[2]	
fieldID  = args[3]	
outfile = args[4]  	




## +++ Load file containing the header of the phenotype file 

if(!file.exists(phenofile)) {
  cat(paste("\n  ERROR (fetch_pheno.R): File", phenofile, "not found.\n\n")) 
  quit("no")
}

if(!file.exists(headerfile)) {
  cat(paste("\n  ERROR (fetch_pheno.R): File", headerfile, "not found.\n\n")) 
  quit("no")
} else {
  fields = scan(headerfile, character(), quiet = TRUE)
} 

cat(paste("\n  Header with", length(fields), "entries read.\n\n"))




## +++ Get column numbers for the field ID 

searchterm = paste("_", fieldID, "_", sep = "")
ind = which(grepl(searchterm, fields))  	
nr_found = length(ind)				

if(nr_found == 0) {
  cat(paste("\n  PROBLEM (fetch_pheno.R): No entries for the field ID", fieldID, "found in", phenofile, ".\n\n")) 
  quit("no")
} else {
  cat(paste("  Fields found:", paste(fields[ind], collapse = "  "), "\n\n"))   
}

colsToKeep = c(1,ind)  	



## +++ Fetch the phenotypes for this field ID:  

cat("  Fetching phenotype: \n\n") 
pheno1 = fread(phenofile, check.names = TRUE, sep = "\t", showProgress = TRUE, select = colsToKeep) 
pheno1 = as.data.frame(pheno1)    
cat("\n  Done. \n\n") 




## +++ Average if length(ind) > 1   

if(nr_found > 1) {
  cat(paste("  Calculating average of the", nr_found, "fields found: \n"))      
  phenovalues = rowMeans(pheno1[,2:ncol(pheno1)], na.rm = TRUE) 
  cat("  Done. \n\n")  
} else {
  phenovalues = pheno1[,2]   
}




## +++ Reformat to format requested by plink
  
sampleIDs = pheno1[,1]       

cat("  ID's look like that:\n\n")   
head(sampleIDs)
cat("\n\n")

rm(pheno1)  
  
outframe = data.frame(FID = sampleIDs, IID = sampleIDs, phenovalues =  phenovalues)   
colnames(outframe) = c("#FID", "IID", paste0("field_", fieldID))   
 

## +++ Save results to outfile

cat(paste("  Saving to:", outfile, "\n"))  
write.table(outframe, file = outfile, quote = FALSE, row.names = FALSE, col.names = TRUE, sep = "\t" )  
cat("  Done. \n\n") 


