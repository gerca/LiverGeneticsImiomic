#!/usr/bin/env bash


# uwemenzel@gmail.com  

 
## === Clean after GCTA-COJO  

# called by "cojo_pheno.sh"
#  
# can be run without sbatch
#          cojo_clean.sh  --id ${ident}   --phenoname ${phenoname}  --cstart ${cstart}  --cstop ${cstop}   




 
## +++ Hardcoded settings & defaults  

# shopt -s nullglob   deactivated for the ls command below!! 

# no parameters necessary 






## +++ Command line parameters:

prog=$( basename "$0" )

if [ "$#" -lt 8 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -i|--id  <string>          no default" 
  echo "         -pn|--phenoname <string>   no default"  
  echo "         --cstart <chrom>           no default"
  echo "         --cstopt <chrom>           no default"          
  echo ""
  exit 1
fi


while [ "$#" -gt 0 ]
do
  case $1 in
      -i|--id)
          ident=$2
          shift
          ;;
      -pn|--phenoname)
          phenoname=$2
          shift
          ;;	  	  	   
      --cstart)
          cstart=$2
          shift
          ;;
      --cstop)
          cstop=$2
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

chromosomes=$( seq ${cstart} ${cstop} ) 






## +++ Check if the variables are defined  

to_test=(ident phenoname cstart cstop   )

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (cojo_clean.sh): Mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done





## +++ Remove files:

summary_file="${ident}_${phenoname}_cojo.ma"  # cojo input   OBS!! filename also used in cojo_pheno.sh 
rm -f ${summary_file} 

echo ""
echo "  File ${summary_file} removed. (input file for gcta-cojo)"	
echo ""



for chrom in  ${chromosomes[*]}     
do
  out_prefix="${ident}_${phenoname}_cojo_chr${chrom}"  	
  cma_file="${out_prefix}.cma.cojo"   	    		  
  rm -f ${cma_file}
  echo "  File ${cma_file} removed."  
done
echo ""





## +++ Report errors / warnings from all cojo logfiles:  

num_errors=$( grep -i error ${ident}_${phenoname}_cojo_chrom*.log | wc -l )  # LIV_MULT3_liv3_cojo_chrom2.log
num_warnings=$( grep -i warning ${ident}_${phenoname}_cojo_chrom*.log | wc -l )

if [ "${num_errors}" -gt 0 ];then
  echo ""
  err_file="${ident}_${phenoname}_cojo_error.log"  
  echo "  Error messages have been detected in the cojo logfiles, see \"${err_file}\"."
  grep -i error ${ident}_${phenoname}_cojo_chrom*.log > ${err_file} 
  echo "" 
fi

if [ "${num_warnings}" -gt 0 ];then
  echo ""
  warn_file="${ident}_${phenoname}_cojo_warn.log"  
  echo "  Warnings have been detected in the cojo logfiles, see \"${warn_file}\"."
  grep -i warning ${ident}_${phenoname}_cojo_chrom*.log > ${warn_file} 
  echo "" 
fi





## +++ Remove warnings regarding duplicate SNP ID's from logfiles   LIV_MULT3_liv1_cojo_chrom2.log

for chrom in  ${chromosomes[*]}     
do
  prune_log="${ident}_${phenoname}_cojo_chrom${chrom}.log"   # see cojo_pheno.sh: prune_log="${ident}_${phenoname}_cojo_chrom${chrom}.log"
  if [ -s "${prune_log}" ];then  
    nr_dup=$( grep "Duplicated SNP ID" ${prune_log} | wc -l | awk '{print $1}' )  
    if [ "${nr_dup}" -gt 0 ]; then
      rnum=$(( 1 + RANDOM%10000 ))   # avoid interference with jobs running parallel
      grep -v "Duplicated SNP ID" ${prune_log} > ${ident}_${phenoname}_temp_clean_${rnum}.log
      mv {ident}_${phenoname}_temp_clean_${rnum}.log ${prune_log} 
      echo "  ${nr_dup} warnings regarding duplicate variant names removed from ${prune_log}" 
    fi
  else
    echo "  File ${prune_log} does not exist (probably no significant markers for this chromosome)."
  fi
done
echo ""





## +++ gzip log files and stuff 

nr_2zip=$( ls ${ident}_${phenoname}_cojo_chr{1..22}.jma.cojo  ${ident}_${phenoname}_cojo_chr{1..22}.ldr.cojo ${ident}_${phenoname}_cojo_chr{1..22}.badsnps ${ident}_${phenoname}_cojo_chr{1..22}.freq.badsnps 2>/dev/null | wc -l | awk '{print $1}' )

if [ "$nr_2zip" -gt 0 ];then
  tar_out="${ident}_${phenoname}_cojo_files.tar.gz"
  tar -czf  ${tar_out}  ${ident}_${phenoname}_cojo_chr{1..22}.jma.cojo  ${ident}_${phenoname}_cojo_chr{1..22}.ldr.cojo ${ident}_${phenoname}_cojo_chr{1..22}.badsnps ${ident}_${phenoname}_cojo_chr{1..22}.freq.badsnps 2>/dev/null
  rm -f ${ident}_${phenoname}_cojo_chr{1..22}.jma.cojo  ${ident}_${phenoname}_cojo_chr{1..22}.ldr.cojo ${ident}_${phenoname}_cojo_chr{1..22}.badsnps ${ident}_${phenoname}_cojo_chr{1..22}.freq.badsnps 
  # echo ""
  echo "  ${nr_2zip} files zipped to ${tar_out} (jma, ldr, badsnps, freq.badsnps)"
  echo ""
else
  echo ""
  echo "  Nothing found to zip ( no jma, ldr, badsnps, freq.badsnps files found.)"
  echo ""
fi





## +++ Finish  

echo -n "  "  
date 
echo "  Done." 
echo "" 
 




