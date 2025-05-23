#!/usr/bin/env Rscript


# uwemenzel@gmail.com 




## +++ Transform phenotype distributions




## +++ Libraries, functions

quant_normal <- function(x, k = 3/8) {
  if (!is.vector(x)) stop("A numeric vector is expected for x.")   
  if ((k < 0) || (k > 0.5)) stop("Select the offset within the interval (0,0.5).")   
  n <- length(na.omit(x))
  rank.x <- rank(x, na.last = "keep")
  normalized = qnorm((rank.x - k)/(n - 2*k + 1))
  return(normalized)
}





## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)     

if(length(args) < 2) {
  cat("\n")
  cat("  Usage: transform_phenotypes  <phenotype_file>  <log|sqrt|norm>\n") 
  cat("  The phenotype file must be in the format requested by plink.\n") 
  cat("\n")
  quit("no")
}
 

infile = args[1]
method = args[2]         


if(! method %in% c("log", "sqrt", "norm"))  {
  stop(paste("\n\n  ERROR (transform_phenotypes.R): Second argument must be 'log', 'sqrt', or 'norm'. \n\n"))
}

   
if(!file.exists(infile)) { 
  stop(paste("\n\n  ERROR (transform_phenotypes.R): File '", infile, "' not found. \n\n")) 
} else {
  if(method == "log")  cat(paste("\n  Log-transforming the phenotype file", infile, "\n\n")) 
  if(method == "sqrt") cat(paste("\n  Sqrt-transforming the phenotype file", infile, "\n\n")) 
  if(method == "norm") cat(paste("\n  Normal-transformation of the phenotype file", infile, "\n\n"))   
}  




## +++ Read input file 

phenofile = read.table(infile, header = TRUE, sep = "\t", comment.char = "", stringsAsFactors = FALSE)

if(ncol(phenofile) < 3) { 
  stop(paste("\n\n  The file", infile, "does not seem to be a proper phenotype file.\n  Must at least have 3 columns.\n\n"))  
}

if(colnames(phenofile)[1] != "X.FID" | colnames(phenofile)[2] != "IID") {
  stop(paste("\n\n  The file", infile, "does not seem to be a proper phenotype file.\n  Must start with #FID  IID\n\n"))  
}





## +++ Load input file

indata = phenofile[,-c(1:2)]   

number_phenotypes = ncol(phenofile) - 2
cat(paste("  We have", number_phenotypes, "phenotype(s).\n\n"))


if(method == "log") { 
  if(!all(indata > 0)) stop(paste("\n\n  ERROR (transform_phenotypes.R): File", infile, "contains negative elements.\n\n"))
  outdata = log(indata)
}


if(method == "sqrt") { 
  if(!all(indata > 0)) stop(paste("\n\n  ERROR (transform_phenotypes.R): File", infile, "contains negative elements.\n\n"))
  outdata = sqrt(indata)
}


if(method == "norm") {
  if(number_phenotypes == 1) {  
    outdata = quant_normal(indata)   # QN the residuals! 
  } 
  if(number_phenotypes > 1) {
    outdata = data.frame(matrix(0, ncol = number_phenotypes, nrow = nrow(indata)))  
    colnames(outdata) = colnames(indata)
    for(i in 1:number_phenotypes) outdata[,i] = quant_normal(indata[,i])  
  }
}




## +++ Show header of infile and outfile

cat("  Indata:\n")
str(indata)
cat("\n  Outdata:\n")
str(outdata)
cat("\n")





## +++ Save output file

pheno_out = cbind( phenofile[,c(1:2)], outdata)  
colnames(pheno_out) = c("#FID", "IID", colnames(phenofile)[3:length(colnames(phenofile))])

if(method == "log")  fn = paste(infile, "log_trans", sep = ".")
if(method == "sqrt") fn = paste(infile, "sqrt_trans", sep = ".")
if(method == "norm") fn = paste(infile, "norm_trans", sep = ".")

write.table(pheno_out, file = fn, row.names = FALSE, quote = FALSE, sep = "\t")    
cat(paste("  Output phenotype file:", fn, "\n\n"))



## +++ Distribution plots  

if(number_phenotypes == 1) {
  fn = paste0(infile, "_", method, ".png")
  pdf(NULL)  # otherwise, "Rplots.pdf" is created (!?) 
  png(filename = fn)
  par(mfrow=c(2,1))
  hist(indata, col = "red", main = "Before trafo", breaks = 40, xlab = colnames(phenofile)[3], font.main = 1)
  mtext("original phenotype", side = 3, cex = 0.8, col = "black")   
  hist(outdata, col = "blue", main = "After trafo", breaks = 40, xlab = colnames(phenofile)[3], font.main = 1)
  if(method == "log") mtext("log-transformed", side = 3, cex = 0.8, col = "blue")
  if(method == "sqrt") mtext("sqrt-transformed", side = 3, cex = 0.8, col = "blue")
  if(method == "norm") mtext("norm-transformed", side = 3,cex = 0.8, col = "blue")
  dev.off() 
  cat(paste("  Image with distribution plots saved to", fn, "\n\n"))  
  par(mfrow = c(1,1))
} 


if(number_phenotypes > 1) {
  for(i in 1:number_phenotypes) {  
    fn = paste0(infile, "_", method, "_", i, ".png")
    pdf(NULL)  # otherwise, "Rplots.pdf" is created (!?) 
    png(filename = fn)
    par(mfrow=c(2,1))
    hist(indata[,i], col = "red", main = "Before trafo", breaks = 40, xlab = colnames(phenofile)[2+i], font.main = 1)
    mtext("original phenotype", side = 3, cex = 0.8, col = "black")   
    hist(outdata[,i], col = "blue", main = "After trafo", breaks = 40, xlab = colnames(phenofile)[2+i], font.main = 1)
    if(method == "log") mtext("log-transformed", side = 3, cex = 0.8, col = "blue")
    if(method == "sqrt") mtext("sqrt-transformed", side = 3, cex = 0.8, col = "blue")
    if(method == "norm") mtext("norm-transformed", side = 3,cex = 0.8, col = "blue")
    dev.off() 
    cat(paste("  Image with distribution plots saved to", fn, "\n\n"))  
    par(mfrow = c(1,1))  
  }   
} 


