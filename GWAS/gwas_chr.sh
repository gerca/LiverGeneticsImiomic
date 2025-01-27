#!/usr/bin/env bash 


# uwemenzel@gmail.com  


## === Run a GWAS on a single chromosome   





## +++ Calling:

# called by 'run_gwas.sh' : 
#
#   sbatch -A ${account} -p ${partition}  -t ${time}  -J ${c_ident} -o ${logchr} -e ${logchr}  \
# 	   gwas_chr --gen ${pgen_prefix}  --chr ${chrom}  --id ${ident}  --pheno ${phenopath}  --pname ${phenoname}  \
#          --covar ${covarpath}  --cname ${covarname} --mac ${mac} --mr2 "${machr2}"  --hwe ${hwe_pval}





## +++ Hardcoded settings & and defaults 

setfile=~/gwas_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings (but not all can be overwritten) 
else
  echo ""
  echo "  ERROR (gwas_chr.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi




 
## +++ Command line parameters:

prog=$( basename "$0" )

if [ "$#" -lt 14 ]; then
  echo ""
  echo "  Usage: ${prog}" 
  echo "         -i|--id <string>         no default"   
  echo "         -g|--gen <string>        no default" 
  echo "         -c|--chr <int>           no default" 
  echo "         -p|--pheno <file>        no default" 
  echo "         -pn|--pname <string>     no default"
  echo "         -co|--covar <file>       no default" 
  echo "         -cn|--cname <string>     no default"
  echo "         -h|--hwe  <real>         ${setfile}"  
  echo "         --mac <int>              ${setfile}" 
  echo "	 --mr2 <range>            ${setfile}"
  echo ""
  exit 1
fi


while [ "$#" -gt 0 ]
do
  case $1 in
      -g|--gen)
          pgen_prefix=$2   
          shift
          ;;  
      -c|--chr)
          chrom=$2
          shift
          ;;	  
      -i|--id)
          ident=$2
          shift
          ;;
      -p|--pheno)
          phenofile=$2
          shift
          ;;
      -pn|--pname)
          phenoname=$2
          shift
          ;;
      -co|--covar)
          covarfile=$2
          shift
          ;;
      -cn|--cname)
          covarname=$2
          shift
          ;;
      --mac)
          mac=$2
          shift
          ;;
      --mr2)
          machr2=$2
          shift
          ;;
      -h|--hwe)
          hwe_pval=$2
          shift
          ;;	  	  	  	  	    	  	  
      *)
          echo ""
	  echo "  Invalid argument: $1"
	  echo ""
	  exit 1
          ;;
  esac
  shift
done





## +++ Check if the variables are defined  

to_test=(pgen_prefix chrom ident phenofile phenoname covarfile covarname mac machr2 hwe_pval)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (gwas_chr.sh): mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done




## +++ Check chromosome name  

if [[ ! ${chrom} =~ ^[0-9]+$ ]];then    # autosomes only!       
  echo ""
  echo  "  ERROR (gwas_chr.sh): Chromosome name is not valid: " ${chrom} 
  echo  "  			Correct syntax is e.g. --chr 22"
  echo ""
  exit 1 
fi   




 
## +++ Header:

echo ""
START=$(date +%s) 
echo -n "  "
date 
echo "  Job identifier: " ${ident}
echo "  Starting job for chromosome ${chrom}"
echo "  Genotype input file prefix: ${pgen_prefix}"
echo "  Phenotype file: ${phenofile}"
echo "  Phenotype name(s): ${phenoname}"
echo "  Covariate file: ${covarfile}"
echo "  Covariate name(s): ${covarname}"
echo "  Threshold for minor allele frequency (mac): ${mac}"
echo "  Threshold for Hardy-Weinberg p-value: ${hwe_pval}"
echo "  Mach-r2 imputation quality range: ${machr2}"




## +++ Modules: 

answ=$( module list  2>&1 | grep plink2 )   
if [ -z "$answ" ];then
  echo "  Loadung modules ..."  | tee -a ${log}
  module load bioinfo-tools
  module load ${plink2_version}
  prog=$( which plink2 ) 
  echo "  Using: $prog"  | tee -a ${log}   
fi




## +++ Check availability of input files and genotype files:

psam=${pgen_prefix}".psam"	 
pvar=${pgen_prefix}".pvar"	
pgen=${pgen_prefix}".pgen" 	   

if [ ! -f ${psam} ]; then
  echo ""
  echo "  ERROR (gwas_chr.sh): Input file '${psam}' not found."
  echo ""
  exit 1 
fi  

if [ ! -f ${pvar} ]; then
  echo ""
  echo "  ERROR (gwas_chr.sh): Input file '${pvar}' not found."
  echo ""
  exit 1 
fi  

if [ ! -f ${pgen} ]; then
  echo ""
  echo "  ERROR (gwas_chr.sh): Input file '${pgen}' not found."
  echo ""
  exit 1 
fi    

echo "  All required genotype files (.pgen, .pvar, .psam) are available." 

if [ ! -f ${phenofile} ]; then
  echo ""
  echo "  ERROR (gwas_chr.sh): Input file '${phenofile}' not found."
  echo ""
  exit 1 
else
  num_samples=$( wc -l ${phenofile} | awk '{print $1}' )
  num_samples=$(( ${num_samples} - 1 ))
  echo "  Phenotype file available, with ${num_samples} samples."   
fi 

if [ ! -f ${covarfile} ]; then
  echo ""
  echo "  ERROR (gwas_chr.sh): Input file '${covarfile}' not found."
  echo ""
  exit 1 
else
  num_samples=$( wc -l ${covarfile} | awk '{print $1}' )
  num_samples=$(( ${num_samples} - 1 ))
  echo "  Covariates file available, with ${num_samples} samples." 
  clist=$( echo $covarname | sed 's/,/ /g' )
  for name in  ${clist[*]}  
  do  
    indicator=$( head -1 ${covarfile} | grep ${name} | wc -l )
    if [ $indicator -ne 1 ]; then
      echo ""
      echo "  ERROR (gwas_chr.sh): Covariate file '${covarfile}' does not contain the column '${name}'"
      echo ""
      exit 1   
    fi      
  done
fi

echo "" 




## +++ Run plink2 (on a single chromosome)      

echo "  Genotype files for chromosome $chrom :"       
ls -l ${psam}
ls -l ${pvar}  
ls -l ${pgen}  
echo ""

outfile_prefix="${ident}_gwas_chr"${chrom}   


plink2 --glm hide-covar 'cols=chrom,pos,ref,alt1,a1freq,beta,se,p,nobs' \
   --pfile ${pgen_prefix} \
   --pheno ${phenofile} --pheno-name ${phenoname}\
   --covar ${covarfile} --covar-name ${covarname} \
   --no-psam-pheno \
   --covar-variance-standardize \
   --mac ${mac} \
   --hwe ${hwe_pval} \
   --mach-r2-filter ${machr2} \
   --out ${outfile_prefix}


# + Outfiles:   

pname=$( echo $phenoname | tr -s ',' '\t' )  
phenoarray=($pname)   
echo ""
echo " Number of elements in phenoname: ${#phenoarray[*]}" 
echo ""
echo "  Regression results: "

for pname in  ${phenoarray[*]} 
do
  echo ""
  echo "  Phenoname: $pname" 
  echo -n "    "
  out_glm=${outfile_prefix}"."${pname}".glm.linear"  
  ls -l ${out_glm}  
  entries=$( wc -l ${out_glm} | awk '{print $1}' )
  entries=$(( ${entries} - 1 ))
  echo "    Number of entries in output file (.glm.linear) for phenoname \"${pname}\": ${entries}"
  num_NA=$( cat ${out_glm} | awk '{print $NF}' | grep NA | wc -l )
  echo "    Number of entries with unassigned (NA) p-values: ${num_NA}"
done
echo ""


out_logf=${outfile_prefix}".log"
rm -f ${out_logf}    
echo "" 




## +++ Finish 
 
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "  Run time: $DIFF seconds" 
echo "" 
echo -n "  "  
date 
echo "  Done." 
echo "" 



