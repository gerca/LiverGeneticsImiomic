#!/usr/bin/env Rscript


# uwemenzel@gmail.com 



# this script is called by "cojo_collect.sh":  
#
#   cojo_allele   ${summary_file}   ${signif_file}  ${signif_fixed}  


 			



## +++ Libraries, Functions, Auxiliary Files: 

library(data.table)  





## +++ Gommand line arguments    

args = commandArgs(trailingOnly = TRUE)

if(length(args) < 3) {
  cat("\n")
  cat("  Usage:   cojo_allele.R   <infile1>       <infile2>         <outfile> \n")  
  cat("  Example: cojo_allele.R  LIV6_cojo.ma   LIV6_cojo.jma   LIV6_cojo.ma.fixeded \n")
  cat("\n")
  quit("no")
}

cojo_input  = args[1]      
cojo_output = args[2]      
cojo_fixed  = args[3]    




## +++ Check input files:

if(!file.exists(cojo_input))  stop(paste("\n\n  ERROR (cojo_allele.R): File '", cojo_input, "' not found.\n\n"))
if(!file.exists(cojo_output)) stop(paste("\n\n  ERROR (cojo_allele.R): File '", cojo_output, "' not found.\n\n"))





## +++ Read input files:

cojo_out = read.table(cojo_output, header = TRUE, sep = "\t", stringsAsFactors = FALSE)  

marker_ids = c("ID", cojo_out$ID)  # ID to get the header in the output 
mark_temp = paste("markers", sample(1:10000,1), "temp.txt", sep = "_")
write.table(marker_ids, file = mark_temp, quote = FALSE, col.names = FALSE, row.names = FALSE)   
cmd = paste("fgrep -f", mark_temp , cojo_input)    #
cojo_in = fread(cmd = cmd, nThread = 16, header = TRUE, check.names = FALSE, sep = "\t", showProgress = FALSE, stringsAsFactors = FALSE)

invisible(file.remove(mark_temp))

merged = merge(cojo_in, cojo_out, by.x = c("ID", "A1"), by.y = c("ID", "A1"))  
merged = merged[,c(1,9,10,3,2,12,13,14,15,16)]
merged = merged[order(merged$CHR, merged$POS),]  # sort for chr, then pos   
colnames(merged) = c("ID", "CHR", "POS", "OTHER", "A1",	"A1_FREQ", "OBS_CT", "BETA", "SE", "P")


## +++ Save to cojo_fixed 

write.table(merged, file = cojo_fixed, quote = FALSE, row.names = FALSE, col.names = TRUE, sep = "\t")

if(!file.exists(cojo_fixed)) {
  stop(paste("\n\n  cojo_allele.R:  File '", cojo_fixed, "' not written.\n\n"))
} else {
   cat(paste("\n\n  cojo_allele.R:  File '", cojo_fixed, "' succesfully created.\n\n"))
}  




