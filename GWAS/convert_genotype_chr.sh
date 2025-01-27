#!/usr/bin/env bash    



# uwemenzel@gmail.com 



## === Convert genotype files, for a single single chromosome





## +++ Calling:

# this script is called by 'convert_genotype.sh' 
# 
# sbatch -A ${account}  -p ${partition}  -t ${time}  -J ${c_ident} -o ${logchr} -e ${logchr}  \ 
# 	 convert_genotype_chr  --genoid ${genoid}  --chr ${chrom}"  





## +++ Hardcoded settings & and defaults 

setfile=~/convert_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings  
else
  echo ""
  echo "  ERROR (convert_genotype_chr.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi






## +++ Get command line parameters:

prog=$( basename "$0" )

if [ "$#" -lt 4 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -g|--genoid <string>           no default"
  echo "         -c|--chr <int>[-<int>]         1-22"
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

to_test=(genoid chrom  plink2_version infolder outfolder)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (convert_genotype_chr.sh): mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done






## +++ Check chromosome name  

if [[ ! ${chrom} =~ ^[0-9]+$ ]];then    # autosomes only!!  TODO: change if sex chromosomes are to be included!       
  echo ""
  echo  "  ERROR (convert_genotype_chr.sh): Chromosome name is not valid: " ${chrom} 
  echo  "                                   Correct syntax is e.g. --chrom 22"
  echo ""
  exit 1 
fi   


 


 
## +++ Header:   (everything going to the sbatch-log: ${logchr}) 

account=$( echo $HOSTNAME | awk 'BEGIN{FS="-"} {print $1}' ) 
echo ""
START=$(date +%s) #  1574946757
echo -n "  "
date
echo "  Account: ${account}" 
echo -n "  Operated by: " 
whoami  
echo "  plink2 version number: ${plink2_version}"
echo ""
echo "  Running on chromosome: $chrom"
echo "  Genotype identifier: ${genoid}"
echo "  Genotype input folder: ${infolder}"
echo "  Genotype output folder: ${outfolder}"
echo "" 
 




## +++ Modules: 

answ=$( module list  2>&1 | grep plink2 )   
if [ -z "$answ" ];then
  echo "  Loadung modules ..."  | tee -a ${log}
  module load bioinfo-tools
  module load ${plink2_version}
  prog=$( which plink2 ) 
  echo "  Using: $prog"  | tee -a ${log}   
fi
echo ""




## +++ Run plink2 (on a single chromosome)      

pgen_prefix="${infolder}/${genoid}_chr$chrom"     

psam="${pgen_prefix}.psam"	
pvar="${pgen_prefix}.pvar"	 
pgen="${pgen_prefix}.pgen" 	   

if [ ! -f ${psam} ]; then
  echo "" 
  echo "  ERROR (convert_genotype_chr.sh): Input file '${psam}' not found." 
  echo "" 
  exit 1 
fi  

if [ ! -f ${pvar} ]; then
  echo "" 
  echo "  ERROR (convert_genotype_chr.sh): Input file '${pvar}' not found." 
  echo "" 
  exit 1 
fi  

if [ ! -f ${pgen} ]; then
  echo "" 
  echo "  ERROR (convert_genotype_chr.sh): Input file '${pgen}' not found." 
  echo "" 
  exit 1 
fi    


echo "  Genotype files for chromosome $chrom :"   # psam, pvar, pgen defined above    
ls -l ${psam}
ls -l ${pvar}  
ls -l ${pgen}  
echo ""
 

outfile_prefix="${outfolder}/${genoid}_chr${chrom}"  # "MF_chr22"
 
echo "plink2 --pfile ${pgen_prefix} --make-bed --out ${outfile_prefix}" 
      plink2 --pfile ${pgen_prefix} --make-bed --out ${outfile_prefix}      

# output:
plinklog="${outfile_prefix}.log" 
bedfile="${outfile_prefix}.bed"
bimfile="${outfile_prefix}.bim"
famfile="${outfile_prefix}.fam"




## +++ Finish  

rm -f ${plinklog}   # batchlog is sufficient 
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "  Run time: $DIFF seconds" 
echo "" 
echo "  .bed file (genotype): ${bedfile}"
echo "  .bim file (markers): ${bimfile}"
echo "  .fam file (samples): ${famfile}"
echo ""
echo -n "  "  
date 
echo "  Done." 
echo "" 






