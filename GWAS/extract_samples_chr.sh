#!/usr/bin/env bash    


# uwemenzel@gmail.com 


## === Extract samples from the genotype files, for a single chromosome





## +++ Call:
#
# called by extract_samples.sh 
#
# sbatch -A ${account} -p ${partition} -t ${time} -J ${c_ident} -o ${logchr} -e ${logchr} \
#        extract_samples_chr --genoid ${genoid} --new_genoid ${new_genoid} --gtype ${gtype} --sfile ${sfile} --chr ${chrom} 

# 
# sbatch -A sens2019016 -p node -t 15:00 -J XTRCT-17 -o extract_chr17.log -e extract_chr17.log \
#         extract_samples_chr --genoid MF --new_genoid MF2 --gtype PGEN  --sfile MRI_IDs_21.txt --chr 17
#
#   --gtype must be PGEN or BED 




## +++ Hardcoded settings & and defaults 

setfile=~/extract_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings  
else
  echo ""
  echo "  ERROR (extract_samples_chr.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi




 
## +++ Command line parameters (override the settings in $setfile):

prog=$( basename "$0" )

if [ "$#" -lt 10 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -g|--genoid <string>           no default"
  echo "         -n|--new_genoid <string>       no default"  
  echo "         -t|--gtype                     no default"
  echo "         -s|--sfile <file>              no default"  
  echo "         -c|--chr <int>                 no default"  
  echo ""
  exit 1
fi

while [ "$#" -gt 0 ]
do
  case $1 in
      -g|--genoid)
          genoid=$2    
          shift
          ;;
      -n|--new_genoid)
          new_genoid=$2    
          shift
          ;;	  
      -t|--gtype)
          gtype=$2    
          shift
          ;;	  
      -s|--sfile)
          sfile=$2    
          shift
          ;;	    
      -c|--chr)
          chrom=$2
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




## +++ Check if the variables are defined (including those defined in the settings file)    

to_test=(genoid new_genoid sfile chrom plink2_version genofolder gtype)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (extract_samples_chr.sh): mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done





## +++ Check chromosome name  

if [[ ! ${chrom} =~ ^[0-9]+$ ]];then    # autosomes only!       
  echo ""
  echo  "  ERROR (extract_samples_chr.sh): Chromosome name is not valid: " ${chrom} 
  echo  "  	            Correct syntax is e.g. --chr 22"
  echo ""
  exit 1 
fi   



 
## +++ Header:

START=$(date +%s)      
echo -n "  "  
date | tee -a ${log}
account=$( echo $HOSTNAME | awk 'BEGIN{FS="-"} {print $1}' ) 
echo "  Account: ${account}"  
echo -n "  Operated by: " 
whoami  
echo "" 
echo "  Running on chromosome $chrom"   
echo "  Genotype input folder: ${genofolder}"  
echo "  Genotype identifier:  ${genoid}"
echo "  New genotype identifier:  ${new_genoid}" 
echo "  Samples to extract: ${sfile}"  
echo "" 
echo "  Requested partition: ${partition}" 
echo "  Requested runtime per chromosome: ${minutes} minutes." 
echo "" 




## +++ Modules: 

answ=$( module list  2>&1 | grep plink2 )   
if [ -z "$answ" ];then
  echo "  Loadung modules ..."  
  module load bioinfo-tools
  module load ${plink2_version}
  prog=$( which plink2 ) 
  echo "  Using: $prog"     
fi
echo ""





## +++ Run plink2 (one chromosome)  

# Input filtering:  https://www.cog-genomics.org/plink/1.9/filter


# PGEN

if [ "${gtype}" == "PGEN" ];then
  pfile_prefix="${genofolder}/${genoid}_chr${chrom}"   
  outfile_prefix="${genofolder}/${new_genoid}_chr${chrom}"
  plink2 --pfile ${pfile_prefix} --keep ${sfile} --make-pgen --out ${outfile_prefix}
  echo ""
  ls -l ${outfile_prefix}.pgen  ${outfile_prefix}.pvar  ${outfile_prefix}.psam
  echo ""  
  rm -f ${outfile_prefix}.log  
fi


# BED

if [ "${gtype}" == "BED" ];then 
  bfile_prefix="${genofolder}/${genoid}_chr${chrom}"   
  outfile_prefix="${genofolder}/${new_genoid}_chr${chrom}"
  plink2 --bfile ${bfile_prefix} --keep ${sfile} --make-bed --out ${outfile_prefix}
  echo ""
  ls -l ${outfile_prefix}.bed  ${outfile_prefix}.bim  ${outfile_prefix}.fam 
  echo "" 
  rm -f ${outfile_prefix}.log # everything in batch log 
fi





## +++ Finish 
 
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "  Run time: $DIFF seconds" 
echo "" 
echo -n "  "  
date 
echo "  Done." 
echo "" 



