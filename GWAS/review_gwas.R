#!/usr/bin/env Rscript


# uwemenzel@gmail.com 




## +++ View GWAS results (html) 




# +++ Calling:
#
# this script is called by "review_gwas.sh":   
#
#   sbatch -A ${account} -p ${partition}  -t ${time}  -J ${ident} -o ${batchlog} -e ${batchlog}  \
# 	   --wrap="module load R_packages/3.6.1; review_gwas.R  ${ident}  ${phenoname}  ${cstart}  ${cstop}" 
#                                            





## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)     


if(length(args) < 4) {
  cat("\n")
  cat("  Insufficient number of arguments\n")
  cat("  Usage: review_GWAS  <jobid>  <phenoname>  <from_chr>  <to_chrom> \n")  
  cat("  Example: review_GWAS  LIV6  1 22 \n")
  cat("\n")
  quit("no")
}

ident = args[1]   
phenoname = args[2]  
cstart = as.integer(args[3])      
cstop = as.integer(args[4]) 
chromosomes = seq(cstart, cstop)  




## +++ Hardcoded settings & and defaults 

setfile = "~/review_settings.R"
if(file.exists(setfile))   {
  source(setfile) 
} else {
  stop(paste("\n\n  ERROR (review_gwas.R): Settings file", setfile, "not found.\n\n"))
}




## +++ Get environment variable 

scripts = Sys.getenv("SCRIPT_FOLDER") 
if ( scripts == "" ) {
  stop(paste("\n\n  ERROR (review_gwas.R): Environment variable 'SCRIPT_FOLDER' not set.\n\n"))  
} else {
  cat(paste("\n  Environment variable 'SCRIPT_FOLDER' is set to", scripts, "\n"))
}




## +++ Libraries, Functions, Auxiliary functions: 

suppressWarnings(library(data.table))   
suppressWarnings(library(WriteXLS))
suppressWarnings(library(rmarkdown))  
# library(qqman)  # alternative for Manhattan and QQ-plot 


# Auxiliary functions

review_functions = paste(scripts, "review_functions.R", sep="/")   
if(file.exists(review_functions)) {
  source(review_functions)
} else {
  stop(paste("\n\n  ERROR (review_gwas.R): File with auxiliary functions '", review_functions, "' not found.\n\n"))
}

gentab_file = paste(scripts, "XM_genes.RData", sep="/")
if(!file.exists(gentab_file)) stop(paste("\n\n  ERROR (review_gwas.R): File", gentab_file, "not found.\n\n"))



# Rmarkdown template :

rmd = paste(scripts, "gwas_report.Rmd", sep="/")
if(!file.exists(rmd)) stop(paste("\n\n  ERROR (review_gwas.R): File ",  rmd,  " not found.\n\n"))

# copy the .Rmd file to the current folder ==> working directory for knitr
if(!file.copy(rmd, getwd())) stop(paste("\n\n  ERROR (review_gwas.R): Could not copy file ",  rmd,  " to the current folder.\n\n"))
rmd = paste(getwd(), "gwas_report.Rmd", sep="/")
if(!file.exists(rmd)) stop(paste("\n\n  ERROR (review_gwas.R): File ",  rmd,  " not copied to current location.\n\n"))


cat(paste("  Rmd file is:", rmd, "\n\n"))   

## +++ Read GWAS parameters: 

parfile = paste(ident, "_gwas_params.txt", sep="")     
if(!file.exists(parfile)) stop(paste("\n\n  ERROR (review_gwas.R) : Could not find the GWAS parameter file: '", parfile, "'.\n\n"))

parameters = readLines(parfile)
parameters = strsplit(parameters, "\\s+") 

# entries with 2 columns:
to_assign = c("plink2_version", "genotype_id", "covarname", "mac", "hwe_pval", "machr2_low", "machr2_high")
for (var in to_assign) {
  row = unlist(parameters[grep(var, parameters)]) 
  if(length(row) == 0) stop(paste("\n\n  ERROR (review_gwas.R): The mandatory entry '", var, "' is missing in the parameter file '", parfile, "'.\n\n")) 
  assign(row[1], row[2])
}

to_assign = c("cojo_out", "clump_out")
for (var in to_assign) {
  rows = parameters[grep(var, parameters)]   
  if(length(rows) > 0) for (i in 1:length(rows)) if(rows[[i]][1] == var & rows[[i]][2] == phenoname) assign(var, rows[[i]][3])  
}

 


## +++ Create header for sbatch logfile 

cat("\n")
date()
getwd()				
R.Version()$platform		
R.Version()$version.string	
cat("\n")
cat(paste("  GWAS JobID:", ident, "\n"))
cat(paste("  Phenotype name:", phenoname, "\n"))
cat(paste("  Considering chromosomes", cstart, "to", cstop, "\n"))
cat(paste("  Bandwidth for kernel density plot of beta:", bandwidth, "\n"))
cat(paste("  Significance threshold (p-values):", p_threshold, "\n"))
if(exists("cojo_out")) cat(paste("  GCTA-COJO output file:", cojo_out, "\n\n")) else cat("\n  GCTA-COJO has not been conducted.\n")
if(exists("clump_out")) cat(paste("  Clump output file:", clump_out, "\n\n")) else cat("\n  Clumping has not been conducted.\n")
cat("\n") 


gwas = data.frame(CHROM = integer(), POS = integer(), ID = character(), REF = character(), ALT1 = character(), A1 = character(), 
		  A1_FREQ = numeric(), OBS_CT = integer(),  BETA = numeric(), SE = numeric(), P = numeric(), stringsAsFactors = FALSE) 

		  		  
for (chr in chromosomes)  {			      
  
  cat(paste("\n  === Collecting results for chromosome", chr, "===\n"))    
  regression_output = paste0(ident, "_gwas_chr", chr, ".", phenoname, ".glm.linear" )  # name determined in "gwas_chr.sh"    
  if(!file.exists(regression_output)) stop(paste("\n\n  ERROR (review_gwas.R) : Could not find the GWAS regression results: '", regression_output, "'.\n\n")) 
  gwas_chr = fread(regression_output, nThread = 16, header = TRUE, check.names = FALSE, sep = "\t", showProgress = FALSE, stringsAsFactors = FALSE)
  	
  gwas = rbind(gwas, gwas_chr, use.names = FALSE)  
     
  ## Logfile info:
  cat(paste("  Regression output:", regression_output, "\n"))  		 
  cat(paste("  Number of markers in the regression output:", nrow(gwas_chr), "\n"))
  num_NA = sum(is.na(gwas_chr$P))
  cat(paste("  Number of unassigned p-values in the regression output:", num_NA, "\n"))
  num_signif = sum(gwas_chr$P < 5.0e-8, na.rm = TRUE)	
  cat(paste("  Number of significant markers (5.0e-8) in the regression output:", num_signif, "\n")) 
}

rm(gwas_chr) 
  
colnames(gwas) = c("CHROM", "POS", "ID", "REF", "ALT1", "A1", "A1_FREQ", "OBS_CT", "BETA", "SE", "P")
cat(paste("\n\n  Total number of markers in the global regression output:", nrow(gwas), "(all chromosomes)\n"))




## +++ Store all significant markers across the genome in .RData object:

sigmarkers = gwas[which(gwas$P <= p_threshold),]       

other_allele = ifelse(sigmarkers$A1 == sigmarkers$REF, sigmarkers$ALT1, sigmarkers$REF) 
sigmarkers = cbind(sigmarkers, other_allele)     
sigmarkers = as.data.frame(sigmarkers[,c(3,1,2,12,6,7,8,9,10,11)])    
colnames(sigmarkers) = c("ID", "CHR", "POS", "OTHER", "A1", "A1_FREQ", "OBS_CT", "BETA", "SE", "P")
signif_file = paste(ident, phenoname, "signif_markers.RData", sep = "_")    
save(sigmarkers, file = signif_file)
cat(paste("  Significant (unpruned) markers saved to", signif_file, "\n"))




## +++ Store all cojoed markers in .RData object (if cojo was conducted):

if(exists("cojo_out")) {   
  cojo_file = paste(ident, phenoname, "cojoed_markers.RData", sep = "_")   	    
  cojo_orig_file = paste(ident, phenoname, "cojoed_orig_markers.RData", sep = "_") 
  cojoed_markers = read.table(cojo_out, header = TRUE, stringsAsFactors = FALSE)   
  save(cojoed_markers, file = cojo_file)  
  cat(paste("\n  Cojoed markers saved to", cojo_file, "\n"))  
  # add the "original" values of the pruned markers, too (original beta, se, obs_ct, p value etc)
  cojoed_markers_orig = sigmarkers[which(sigmarkers$ID %in% cojoed_markers$ID),] 
  save(cojoed_markers_orig, file = cojo_orig_file)
  cat(paste("  Cojoed markers with original results saved to", cojo_orig_file, "\n"))
} else {
  cat(paste("  It seems that GCTA-COJO was not run. ( No 'cojo_out' entry in", parfile, ").\n"))
}




## +++ Store all clumped markers in .RData object (if clumping was conducted):

if(exists("clump_out")) {   # if we have clump results     LIV_MULT3_liv3_clump.jma
  clump_file = paste(ident, phenoname, "clumped_markers.RData", sep = "_")   	 
  clumped_markers = read.table(clump_out, header = TRUE, stringsAsFactors = FALSE)   
  save(clumped_markers, file = clump_file)  
  cat(paste("  Clumped markers saved to", clump_file,"\n"))  
} else {
  cat(paste("  It seems that clumping was not run. ( No 'clump_out' entry in", parfile, ").\n"))
}
  
  
## +++ Genomic inflation factor, lambda

p = length(gwas$P)  						
expect.stats = qchisq(ppoints(p), df = 1, lower.tail = FALSE)	 
obs.stats = qchisq(gwas$P, df = 1, lower.tail = FALSE)		 
lambda = median(obs.stats)/median(expect.stats) 		

 


## +++ Excel with unpruned or pruned global significant markers     

excel_file = paste(ident, phenoname, "signif_markers.xls", sep = "_")  

if(exists("cojo_out") & exists("clump_out")) {   	 
  if((nrow(cojoed_markers) != 0) | (nrow(clumped_markers) != 0)) { 
    WriteXLS(c("cojoed_markers", "cojoed_markers_orig", "clumped_markers"), ExcelFileName = excel_file, SheetNames = c("cojoed", "orig_cojo", "clumped"), row.names = FALSE)
    cat(paste("  Spreadsheet with cojoed and clumped markers saved to", excel_file, ".\n\n"))
  }
}

if(exists("cojo_out") & !exists("clump_out")) { 	
  if(nrow(cojoed_markers) != 0) { 
    WriteXLS(c("cojoed_markers", "cojoed_markers_orig"), ExcelFileName = excel_file, SheetNames = c("cojoed", "orig_cojo"), row.names = FALSE)
    cat(paste("  Spreadsheet with cojoed markers saved to", excel_file, ".\n\n"))
  }
}

if(!exists("cojo_out") & exists("clump_out")) { 
  if(nrow(clumped_markers) != 0) { 
    WriteXLS(c("clumped_markers"), ExcelFileName = excel_file, SheetNames = c("clumped"), row.names = FALSE)
    cat(paste("  Spreadsheet with clumped markers saved to", excel_file, ".\n\n"))
  }
}

if(!exists("cojo_out") & !exists("clump_out")) { 	
  if(nrow(sigmarkers) != 0) {
    WriteXLS(sigmarkers, ExcelFileName = excel_file, SheetNames = ident, row.names = FALSE)
    cat(paste("  Spreadsheet with significant (unpruned) markers saved to", excel_file, ".\n\n"))
  }         
}




## +++ Find nearest genes for the significant markers

cat("  Looking for nearest genes ...  ") 
start_time = Sys.time()  

gentab = get(load(gentab_file))  

if(exists("cojo_out")) {
  cojo_near_genes = apply(cojoed_markers, 1, link_nearest_gene)
  cojo_neargene_file = paste(ident, phenoname, "cojo_nearest_genes.RData", sep = "_")
  save(cojo_near_genes, file = cojo_neargene_file)
}

if(exists("clump_out")) {
  clump_near_genes = apply(clumped_markers, 1, link_nearest_gene)
  clump_neargene_file = paste(ident, phenoname, "clump_nearest_genes.RData", sep = "_")
  save(clump_near_genes, file = clump_neargene_file) 
}

if(!exists("cojo_out") & !exists("clump_out")) {  
  signif_near_genes = apply(sigmarkers, 1, link_nearest_gene)
  signif_neargene_file = paste(ident, phenoname, "signif_nearest_genes.RData", sep = "_")
  save(signif_near_genes, file = signif_neargene_file)   
}
   
stop_time = Sys.time()
diff_time = stop_time - start_time 
cat(paste("Done in", round(diff_time,2), "seconds.\n"))
 
 
 
 

 
## +++ Global Manhattan plot (all chromosomes)       

cat("  Creating Manhattan plot ...  ") 
start_time = Sys.time()  

manhat = gwas[,c(3,1,2,11)]    
colnames(manhat) = c("marker", "chr", "pos", "pvalue") 


txt = paste("Manhattan plot for jobid", ident) 
man_plot = paste(ident, phenoname, "Manhattan.png", sep = "_") 
png(man_plot, width = 1024, height = 768)

if (annotation) {
  ann = annotateSNPRegions(manhat$marker, manhat$chr, manhat$pos, manhat$pvalue,
	snplist =c("rs1558902","rs17024393"),
	labels = c("FTO","GNAT2"),
	col = c("red","green"), kbaway=50)
  print(manhattan.plot(manhat$chr, manhat$pos, manhat$pvalue, sig.level = 5e-8, col = colvec, should.thin = F, main = txt, annotate = ann)) 
} else {  
  print(manhattan.plot(manhat$chr, manhat$pos, manhat$pvalue, sig.level = 5e-8, col = colvec, should.thin = F, main = txt)) 
}
invisible(dev.off())

stop_time = Sys.time()
diff_time = stop_time - start_time 
cat(paste("Done in", round(diff_time,2), "seconds.\n"))




## +++ Global QQ-plot (all chrom)     

cat("  Creating QQ-plot ...  ") 
start_time = Sys.time()  

txt = paste("Job", ident, ": QQ-plot for -log10 of p-values") 
qq_plot = paste(ident, phenoname, "QQplot.png", sep = "_") 
png(qq_plot, width = 1024, height = 768)
qqplot(-log10(ppoints(nrow(gwas))),-log10(gwas$P), xlab = "theoretical", ylab = "observed", main = "Q-Q Plot for -log10 Pval") 
abline(0, 1, col = "red") 
invisible(dev.off())
 
stop_time = Sys.time()
diff_time = stop_time - start_time 
cat(paste("Done in", round(diff_time,2), "seconds.\n"))



 
## +++ Global histogram of beta-values

cat("  Creating histogram for regression slopes ...  ") 
start_time = Sys.time()  
  
txt = paste(ident, ": Histogram of beta-values") 
histo_plot = paste(ident, phenoname, "beta_hist.png", sep = "_") 
png(histo_plot, width = 1024, height = 768)
hist(gwas$BETA, col = "red", breaks = 50, xlab = "beta", main = txt, font.main = 1) 
invisible(dev.off())
  
stop_time = Sys.time()
diff_time = stop_time - start_time 
cat(paste("Done in", round(diff_time,2), "seconds.\n"))

 
 
## +++ Global kernel density plot of beta-values 
 
cat("  Creating kernel density plot for regression slopes ...  ") 
start_time = Sys.time()  
 
d = density(gwas$BETA, na.rm = TRUE)
txt = paste(ident, ": Kernel density plot for beta")
kernel_plot = paste(ident, phenoname, "beta_kernel.png", sep = "_")
png(kernel_plot, width = 1024, height = 768)
plot(d, main = txt, col = "red", font.main = 1)
polygon(d, col="red", border="darkgrey")  # fill 
rug(jitter(gwas$BETA))
invisible(dev.off())
  
stop_time = Sys.time()
diff_time = stop_time - start_time 
cat(paste("Done in", round(diff_time,2), "seconds.\n"))

 
 


## +++ Create output html from rmarkdown template:  

if(!file.exists(rmd)) 		 stop(paste("\n\n  ERROR (review_gwas.R): Rmarkdown template file", rmd, "not found.\n\n"))
if(!file.exists(signif_file))    stop(paste("\n\n  ERROR (review_gwas.R): File", signif_file, "not found.\n\n")) 
if(!file.exists(qq_plot)) 	 stop(paste("\n\n  ERROR (review_gwas.R): File", qq_plot, "not found.\n\n"))  
if(!file.exists(man_plot)) 	 stop(paste("\n\n  ERROR (review_gwas.R): File", man_plot, "not found.\n\n"))   
if(!file.exists(histo_plot)) 	 stop(paste("\n\n  ERROR (review_gwas.R): File", histo_plot, "not found.\n\n"))  
if(!file.exists(kernel_plot)) 	 stop(paste("\n\n  ERROR (review_gwas.R): File", kernel_plot, "not found.\n\n"))    

covar = unlist(strsplit(covarname, ","))    
covar = paste(covar, collapse = " + ")     
lm_call = paste(phenoname , "~ marker +", covar)    


plist = list()
plist["workfolder"] = getwd()    		
plist["ident"] = ident 
plist["nr_samples"] = unique(gwas$OBS_CT)       
plist["nr_markers"] = nrow(gwas)  
plist["genoid"] = genotype_id
plist["lm_call"] = lm_call      
plist["phenoname"] = phenoname
plist["plink"] = plink2_version  		  
plist["lambda"] = lambda 
plist["mac"] = mac  
plist["machr2_low"] = machr2_low  
plist["hwe_pval"] = hwe_pval      
plist["excel_file"] = excel_file
plist["cojo_done"] = ifelse(exists("cojo_out"), TRUE, FALSE)
plist["clump_done"] = ifelse(exists("clump_out"), TRUE, FALSE)
plist["signif_file"] = signif_file
plist["clump_file"] = ifelse(exists("clump_out"), clump_file, "")  
plist["cojo_file"] = ifelse(exists("cojo_out"), cojo_file, "")
plist["cojo_orig_file"] = ifelse(exists("cojo_out"), cojo_orig_file, "")
plist["cojo_neargene_file"] = ifelse(exists("cojo_out"), cojo_neargene_file, "")
plist["clump_neargene_file"] = ifelse(exists("clump_out"), clump_neargene_file, "")
plist["signif_neargene_file"] = ifelse(!exists("cojo_out") & !exists("clump_out"), signif_neargene_file, "")   
plist["qq_plot"] = qq_plot    
plist["man_plot"] = man_plot
plist["histo_plot"] = histo_plot
plist["kernel_plot"] = kernel_plot

outname = paste(ident, phenoname, "report.html", sep = "_")    
cat(paste("  Rendering file", rmd, " ..."))   
start_time = Sys.time()  

rmarkdown::render(rmd, params = plist, output_dir = getwd(), output_file = outname, quiet = TRUE)  

stop_time = Sys.time()
diff_time = stop_time - start_time 
cat(paste("  Done in", round(diff_time,2), "seconds.\n"))



## +++ Finish

cat(paste("\n  ", date(),"\n\n"))
cat("  Done.\n\n")  


 
 

















 
