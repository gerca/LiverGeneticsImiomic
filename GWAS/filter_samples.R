#!/usr/bin/env Rscript


# uwemenzel@gmail.com 




## +++ Filter a list of samples:
 

## +++ Call:


## PGEN:    
# filter_samples ukb_imp_v3_chr19.psam  a,b  filtered.txt  
# filter_samples ukb_imp_v3_chr19.psam  a,b,c,d  filtered2.txt  
# filter_samples ukb_imp_v3_chr19.psam a,b,c,d,e  filtered_female.txt    
# filter_samples ukb_imp_v3_chr19.psam a,b,c,d,f  filtered_male.txt       

# BED:
# filter_samples  FTD_chr22.fam  a filtered_relat.txt  

# RAW, 2 columns:
# filter_samples  filter_input.txt  c,f  filter_output_1.txt

# RAW, 1 column:
# filter_samples  filter_in_one.txt  c,e  filter_output_2.txt





## +++ Libraries, Functions

getExtension <- function(file){ 
    ex <- strsplit(basename(file), split="\\.")[[1]]
    return(ex[-1])
} 




## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)      

if(length(args) != 3) {
  cat("\n")
  cat("  Usage: filter_samples   <infile>   <options>  <outfile> \n\n") 
  cat("    Options: (comma-separated list): \n") 
  cat("      a: keep only unrelated participants \n")
  cat("      b: keep only caucasians \n")
  cat("      c: keep only MRI-scanned participants \n")    
  cat("      d: remove samples with sex chromosome aneuploidy \n")  
  cat("      e: keep females only \n")  
  cat("      f: keep males only \n")   
  cat("\n")
  quit("no")
}

infile = args[1]    
options = args[2]
outfile = args[3]  




## +++ Infiles & outfiles  

if(!file.exists(infile)) {
  cat(paste("\n  ERROR (filter_samples.R): File", infile, "not found.\n\n")) 
  quit("no")
} 


if(file.exists(outfile)) {
  cat(paste("\n  File", outfile, "already exists.\n")) 
  cat("  Delete the existing file or choose another outfile name.\n\n")
  quit("no")
} 




## +++ Options   

opvector = unlist(strsplit(options, split = ",")) 

if(class(opvector) != "character") {
 cat(paste("\n  ERROR (filter_samples.R): options '", options,  "' appear to have wrong format. \n  Use comma-separated list or a single character.\n\n"))
 quit("no")
}

for (op in opvector) {
   if(nchar(op) != 1) {
     cat(paste("\n  ERROR (filter_samples.R): Option '", op,  "' is not a single character.\n\n"))
     quit("no")
   }   
   if( ! op %in% c("a", "b", "c", "d", "e", "f")) {
     cat(paste("\n  ERROR (filter_samples.R): Option '", op,  "' is not valid. Must be a, b, c, d, e, or f.\n\n"))
     quit("no")
   }  
}

if(("e" %in% opvector) & ("f" %in% opvector)) {
  cat(paste("\n  ERROR (filter_samples.R): It does not make sense to filter for both female and male (options 'e' and 'f').\n\n"))
  quit("no")
}




## +++ Load infile  (to filter)   

line1 = read.table(infile, header = FALSE, sep = "\t", nrow = 1)
found = 0


if(ncol(line1) == 3) { 
  suffix = getExtension(infile) 
  if(suffix != "psam") {
    cat(paste("\n  ERROR (filter_samples.R): The input file should have the suffix ' psam ' not '", suffix, "'.\n\n"))
    quit("no")
  } else {
    cat("\n  Input file is probably a .psam file.\n\n")
  }
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)  
  filtered_samples = indata[,2]  # IID
  if(class(filtered_samples) != "integer") {
    cat(paste("\n  ERROR (filter_samples.R): The 2nd column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(filtered_samples)
    cat("\n\n")
    quit("no")
  } 
  found = found + 1 
}

if(ncol(line1) == 6) {    
  suffix = getExtension(infile)   
  if(suffix != "fam") {
    cat(paste("\n  ERROR (filter_samples.R): The input file should have the suffix ' fam ' not '", suffix, "'.\n\n"))
    quit("no")
  } else {
    cat("\n  Input file is probably a .fam file.\n\n")
  }
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)  
  filtered_samples = indata[,2]  
  if(class(filtered_samples) != "integer") {
    cat(paste("\n  ERROR (filter_samples.R): The 2nd column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(filtered_samples)
    cat("\n\n")
    quit("no")
  } 
  found = found + 1
}

if(ncol(line1) == 2) {  
  cat("\n  Input file is probably a raw file with 2 columns.\n\n")
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)  
  filtered_samples = indata[,2]  
  if(class(filtered_samples) != "integer") {
    cat(paste("\n  ERROR (filter_samples.R): The 2nd column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(filtered_samples)
    cat("\n\n")
    quit("no")
  } 
  found = found + 1
}

if(ncol(line1) == 1) {    
  cat("\n  Input file is probably a raw file with 1 column.\n\n")
  indata = read.table(infile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)  
  filtered_samples = indata[,1]  
  if(class(filtered_samples) != "integer") {
    cat(paste("\n  ERROR (filter_samples.R): The 1st column in", infile, "does not seem to contain sample IDs.\n\n"))
    head(filtered_samples)
    cat("\n\n")
    quit("no")
  }
  found = found + 1 
}


if (found != 1) {
  cat("\n  ERROR (filter_samples.R): The input file type could not be identified.\n")
  cat("  Allowed are '.psam', '.fam', or text files with one or two columns, containing sample IDs.\n\n")
  quit("no")
} else {   
  cat(paste("  The input list contains", length(filtered_samples), "participants.\n\n"))
}




## +++ Load filters 

scripts = Sys.getenv("SCRIPT_FOLDER")  

if( "a" %in% opvector) {
  fn = paste(scripts, "IDs_unrelated.RData", sep="/")  
  if(!file.exists(fn)) {
    cat(paste("\n  ERROR (filter_samples.R): File", fn, "not found.\n\n"))
    quit("no")
  }
  id_unrelated = get(load(fn)) 
  cat("  Applying filter 'a': Keep only unrelated participants. \n")
  cat(paste("    We have", length(id_unrelated), "unrelated individuals available.\n"))   
  filtered_samples = intersect(filtered_samples, id_unrelated)
  if(length(filtered_samples) == 0) {
    cat("  No samples left after this filter. Exiting ...\n\n")
    quit("no")
  }
  cat(paste("    Applying that filter reduces the input list to", length(filtered_samples), "participants.\n\n"))     
} 



if( "b" %in% opvector) {
  fn = paste(scripts, "IDs_caucasian.RData", sep="/") 
  if(!file.exists(fn)) {
    cat(paste("\n  ERROR (filter_samples.R): File", fn, "not found.\n\n"))
    quit("no")
  }  
  id_caucasian = get(load(fn)) 
  cat("  Applying filter 'b': Keep only caucasian participants. \n")  
  cat(paste("    We have", length(id_caucasian), "caucasian individuals available.\n"))   #  
  filtered_samples = intersect(filtered_samples, id_caucasian)
  if(length(filtered_samples) == 0) {
    cat("  No samples left after this filter. Exiting ...\n\n")
    quit("no")
  }  
  cat(paste("    Applying that filter reduces the input list to", length(filtered_samples), "participants.\n\n"))        
}



if( "c" %in% opvector) {   
  fn = paste(scripts, "IDs_MRI.RData", sep="/") 
  if(!file.exists(fn)) {
    cat(paste("\n  ERROR (filter_samples.R): File", fn, "not found.\n\n"))
    quit("no")
  }  
  id_mri = get(load(fn)) 
  cat("  Applying filter 'c': Keep only MRI-scanned participants. \n")  
  cat(paste("    We have", length(id_mri), "MRI-scanned individuals available.\n"))   #  
  filtered_samples = intersect(filtered_samples, id_mri)
  if(length(filtered_samples) == 0) {
    cat("  No samples left after this filter. Exiting ...\n\n")
    quit("no")
  }  
  cat(paste("    Applying that filter reduces the input list to", length(filtered_samples), "participants.\n\n"))        
} 



if( "d" %in% opvector) {  
  fn = paste(scripts, "IDs_aneuploidy.RData", sep="/")
  if(!file.exists(fn)) {
    cat(paste("\n  ERROR (filter_samples.R): File", fn, "not found.\n\n"))
    quit("no")
  }  
  id_aneuploidy = get(load(fn))
  cat("  Applying filter 'd': Remove samples with sex chromosome aneuploidy. \n")
  cat(paste("    We have", length(id_aneuploidy), "individuals with sex chromosome aneuploidy in the database.\n"))
  filtered_samples = setdiff(filtered_samples, id_aneuploidy)
  if(length(filtered_samples) == 0) {
    cat("  No samples left after this filter. Exiting ...\n\n")
    quit("no")
  }  
  cat(paste("    Applying that filter reduces the input list to", length(filtered_samples), "participants.\n\n"))       
}



if( "e" %in% opvector) {    # female    
  fn = paste(scripts, "IDs_female.RData", sep="/") 
  if(!file.exists(fn)) {
    cat(paste("\n  ERROR (filter_samples.R): File", fn, "not found.\n\n"))
    quit("no")
  }  
  id_female = get(load(fn)) 
  cat("  Applying filter 'e': Keep only female participants. \n")  
  cat(paste("    We have", length(id_female), "female participants available.\n"))   #  
  filtered_samples = intersect(filtered_samples, id_female)
  if(length(filtered_samples) == 0) {
    cat("  No samples left after this filter. Exiting ...\n\n")
    quit("no")
  }  
  cat(paste("    Applying that filter reduces the input list to", length(filtered_samples), "participants.\n\n"))        
} 



if( "f" %in% opvector) {      
  fn = paste(scripts, "IDs_male.RData", sep="/") 
  if(!file.exists(fn)) {
    cat(paste("\n  ERROR (filter_samples.R): File", fn, "not found.\n\n"))
    quit("no")
  }  
  id_male = get(load(fn)) 
  cat("  Applying filter 'f': Keep only male participants. \n")  
  cat(paste("    We have", length(id_male), "male participants available.\n"))   #  
  filtered_samples = intersect(filtered_samples, id_male)
  if(length(filtered_samples) == 0) {
    cat("  No samples left after this filter. Exiting ...\n\n")
    quit("no")
  }  
  cat(paste("    Applying that filter reduces the input list to", length(filtered_samples), "participants.\n\n"))        
} 




## +++ Save the filtered_samples :

outframe = data.frame(FID = filtered_samples, IID = filtered_samples)  
write.table(outframe, file = outfile, col.names = FALSE, quote = FALSE, sep = "\t", row.names = FALSE)

cat(paste("  Output saved to:", outfile, "\n\n")) 
cat("  Use the program 'extract_samples' to obtain a genotype dataset based on the filtered samples.\n\n")
quit("no")  # uwemenzel@gmail.com   














