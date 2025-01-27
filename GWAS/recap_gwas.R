#!/usr/bin/env Rscript


# uwemenzel@gmail.com 



## +++ Summarize the GWAS results (no html, lite version)    



# +++ Calling:
#
# this script is called by "recap_gwas.sh":   
#
#     recap_gwas.R   ${ident}  ${pheno}  ${cstart}  ${cstop}  ${pval}  ${file_list}
#
#  no SLURM 



## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)     


if(length(args) < 6) {
  cat("\n")
  cat("  Insufficient number of arguments\n")
  cat("  Usage: recap_gwas.R  <jobid>  <phenoname>  <from_chr>  <to_chrom>  <pval>  <file> \n")  
  cat("  Example: recap_gwas.R  LIV6  1 22  5e-8  LIV_MULT5_liv2_liv8_liv7_files.txt\n")
  cat("\n")
  quit("no")
}

ident = args[1]   
phenoname = args[2]  
cstart = as.integer(args[3])      
cstop = as.integer(args[4]) 
pval = as.numeric(args[5])
file_list = as.character(args[6])   

chromosomes = seq(cstart, cstop) 
important_files = list()  





## +++ Libraries, Functions, Auxiliary functions: 

suppressWarnings(library(WriteXLS))
suppressWarnings(library(data.table))






## +++ Read GWAS parameters: 

parfile = paste(ident, "_gwas_params.txt", sep="")     
if(!file.exists(parfile)) stop(paste("\n\n  ERROR (recap_gwas.R) : Could not find the GWAS parameter file: '", parfile, "'.\n\n"))
system(paste("touch", parfile))
 
parameters = readLines(parfile)
parameters = strsplit(parameters, "\\s+") 

to_assign = c("cojo_out", "clump_out")
for (var in to_assign) {
  rows = parameters[grep(var, parameters)]    
  if(length(rows) > 0) for (i in 1:length(rows)) if(rows[[i]][1] == var & rows[[i]][2] == phenoname) assign(var, rows[[i]][3])  
}

  


cat(paste("\n  Phenotype name:", phenoname, "\n"))


if(exists("cojo_out")) {
  cojo_file = paste0(ident, "_", phenoname, "_cojo.jma")
  if(file.exists(cojo_file)) {
    cojo_frame = read.table(cojo_file, header = TRUE, stringsAsFactors = FALSE)
    cat(paste0("    Cojo results: ", cojo_file, " (", nrow(cojo_frame), " markers)\n"))
    system(paste("touch", cojo_file))
    important_files = append(important_files, cojo_file)   
  } else {
    stop(paste("\n\n  ERROR (recap_gwas.R) : Could not find cojo results: '", cojo_file, "'.\n\n")) 
  } 
} 


if(exists("clump_out")) {
  clump_file = paste0(ident, "_", phenoname, "_clump.jma")
  if(file.exists(clump_file)) {
    clump_frame = read.table(clump_file, header = TRUE, stringsAsFactors = FALSE)
    cat(paste0("    Clump results: ", clump_file, " (", nrow(clump_frame), " markers)\n"))
    system(paste("touch", clump_file))
    important_files = append(important_files, clump_file)   
  } else {
    stop(paste("\n\n  ERROR (recap_gwas.R) : Could not find clump results: '", clump_file, "'.\n\n")) 
  } 
} 


if(!exists("clump_out") & !exists("cojo_out")) { 
  cat("    No pruning was conducted. Collecting significant markers ...\n")  
  signif_frame = data.frame(CHROM = integer(), POS = integer(), ID = character(), REF = character(), ALT1 = character(), A1 = character(), 
		    A1_FREQ = numeric(), OBS_CT = integer(),  BETA = numeric(), SE = numeric(), P = numeric(), stringsAsFactors = FALSE) 
  
  cat(paste("    Chromosome: "))
  for (chr in chromosomes)  {
    regression_output = paste0(ident, "_gwas_chr", chr, ".", phenoname, ".glm.linear" )     
    if(!file.exists(regression_output)) stop(paste("\n\n  ERROR (recap_gwas.R) : Could not find the GWAS regression results: '", regression_output, "'.\n\n")) 
    cmd = paste0("awk -v p=", pval, " '{if($NF <= p) {print $0}}' ", regression_output) 
    signif_frame_chr = suppressWarnings(fread(cmd = cmd, nThread = 16, header = TRUE, check.names = FALSE, sep = "\t", showProgress = FALSE, stringsAsFactors = FALSE)) 
    signif_frame = rbind(signif_frame, signif_frame_chr, use.names = FALSE)  
    cat(paste(chr, " "))
  }
  
  other_allele = ifelse(signif_frame$A1 == signif_frame$REF, signif_frame$ALT1, signif_frame$REF) 
  signif_frame = cbind(signif_frame, other_allele)     
  signif_frame = as.data.frame(signif_frame[,c(3,1,2,12,6,7,8,9,10,11)])    
  colnames(signif_frame) = c("ID", "CHR", "POS", "OTHER", "A1", "A1_FREQ", "OBS_CT", "BETA", "SE", "P")    
  signif_file = paste0(ident, "_", phenoname, "_unpruned.jma")  
  write.table(signif_frame, file = signif_file, quote = FALSE, sep = "\t", row.names = FALSE) 
  cat(paste0("\n    Significant markers: ", signif_file, " (", nrow(signif_frame), " markers)\n"))
  important_files = append(important_files, signif_file) 
  
} 




## +++ Excel with unpruned or pruned global significant markers     

if(exists("cojo_out") & exists("clump_out")) {   	 
  excel_file = paste(ident, phenoname, "clump_cojo.xls", sep = "_")
  if((nrow(cojo_frame) != 0) | (nrow(clump_frame) != 0)) { 
    WriteXLS(c("cojo_frame", "clump_frame"), ExcelFileName = excel_file, SheetNames = c("cojoed", "clumped"), row.names = FALSE)
    cat(paste("    Spreadsheet:", excel_file, "\n"))
    important_files = append(important_files, excel_file)
  }
}

if(exists("cojo_out") & !exists("clump_out")) { 	
  excel_file = paste(ident, phenoname, "cojo.xls", sep = "_")
  if(nrow(cojo_frame) != 0) { 
    WriteXLS(c("cojo_frame"), ExcelFileName = excel_file, SheetNames = c("cojoed"), row.names = FALSE)
    cat(paste("    Spreadsheet:", excel_file, "\n"))
    important_files = append(important_files, excel_file)  
  }
}

if(!exists("cojo_out") & exists("clump_out")) { 	
  excel_file = paste(ident, phenoname, "clump.xls", sep = "_")
  if(nrow(clump_frame) != 0) { 
    WriteXLS(c("clump_frame"), ExcelFileName = excel_file, SheetNames = c("clumped"), row.names = FALSE)
    cat(paste("    Spreadsheet:", excel_file, "\n"))
    important_files = append(important_files, excel_file)  
  }
}

if(!exists("cojo_out") & !exists("clump_out")) { 	
  excel_file = paste(ident, phenoname, "unpruned.xls", sep = "_") 
  if(nrow(signif_frame) != 0) {
    WriteXLS(signif_frame, ExcelFileName = excel_file, SheetNames = ident, row.names = FALSE)
    cat(paste("    Spreadsheet:", excel_file, "\n"))
    important_files = append(important_files, excel_file)   
  }         
}



## +++ Write list with important files

sink(file_list) 
for (file in important_files) cat(file, sep="\n")
sink()








