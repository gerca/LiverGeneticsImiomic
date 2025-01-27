#!/usr/bin/env bash    



# uwemenzel@gmail.com 





## === Fetch a phenotype by UKBB field ID:
#
# (possibly use search_filedID first to find the field number)  


## +++ Call:
#
# fetch_pheno --field  <fieldID>  <outfile>
#   
# fetch_pheno --field 48 --out test_pheno.txt




## +++ Hardcoded settings & and defaults 

setfile=~/fetch_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings  
else
  echo ""
  echo "  ERROR (fetch_pheno.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi



 
## +++ Command line parameters (override the settings in $setfile):

prog=$( basename "$0" )

if [ "$#" -lt 4 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -f|--field <integer>     no default"
  echo "         -o|--out   <filename>    no default"  
  echo ""
  exit 1
fi

while [ "$#" -gt 0 ]
do
  case $1 in
      -i|--field)
          field=$2    
          shift
          ;;
      -o|--out)
          outfile=$2    
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

to_test=(phenofile1  phenofile2  phenofile3  field  outfile)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (fetch_pheno.sh): mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done



## +++ Check field ID 

if [[ ! ${field} =~ ^[0-9]+$ ]];then          
  echo ""
  echo  "  ERROR (fetch_pheno.sh): Field identifier is not valid: " ${field} 
  echo  "  	                   Must be integer number"
  echo ""
  exit 1 
fi   



## +++ Check availability of input file, and if the outfile already exists:

if [ ! -f ${phenofile1} ]; then
  echo ""    
  echo "  ERROR (fetch_pheno.sh): Phenotype file '${phenofile1}' not found." 
  echo "" 
  exit 1 
fi


if [ ! -f ${phenofile2} ]; then
  echo ""    
  echo "  ERROR (fetch_pheno.sh): Phenotype file '${phenofile2}' not found." 
  echo "" 
  exit 1 
fi


if [ ! -f ${phenofile3} ]; then
  echo ""    
  echo "  ERROR (fetch_pheno.sh): Phenotype file '${phenofile3}' not found." 
  echo "" 
  exit 1 
fi


if [ -f ${outfile} ]; then
  echo ""    
  echo "  WARNING (fetch_pheno.sh): File '${outfile}' already exists. Please choose another filename." 
  echo "" 
  exit 1 
fi



## +++ Modules: 

answ=$( module list  2>&1 | grep R_packages )   
if [ -z "$answ" ];then
  echo ""
  echo -n "  Loadung R modules ..."  
  module load R_packages/3.6.1 
  echo "  Done."
  echo ""
fi



rnum=$(( 1 + RANDOM%10000 ))
headerfile1="ukb23907_header_${rnum}.txt"
head -1 ${phenofile1} > ${headerfile1}  


rnum=$(( 1 + RANDOM%10000 ))
headerfile2="r40779_header_${rnum}.txt"
head -1 ${phenofile2} > ${headerfile2}  


rnum=$(( 1 + RANDOM%10000 ))
headerfile3="r41128_header_${rnum}.txt"
head -1 ${phenofile3} > ${headerfile3}  



# Look which header contains the desired field number    

nr_found=0

tag=$( grep _${field}_ ${headerfile1} ) 
if [ "${tag}" != "" ];then
  nr_found=$((${nr_found} + 1))
  phenofile=${phenofile1}
  headerfile=${headerfile1}
fi

tag=$( grep _${field}_ ${headerfile2} ) 
if [ "${tag}" != "" ];then
  nr_found=$((${nr_found} + 1))
  phenofile=${phenofile2}  
  headerfile=${headerfile2}
fi

tag=$( grep _${field}_ ${headerfile3} ) 
if [ "${tag}" != "" ];then
  nr_found=$((${nr_found} + 1))
  phenofile=${phenofile3}
  headerfile=${headerfile3}
fi


if [ "${nr_found}" -eq 0 ];then
  echo "" 
  echo "  ERROR (fetch_pheno.sh): The field ID ${field} could not be found." 
  echo ""  
  exit 1  
fi


if [ "${nr_found}" -eq 2 ];then
  echo ""
  echo "  ERROR (fetch_pheno.sh): The field ID ${field} occurs in 2 data files."  
  echo "                   Don't know which to choose." 
  echo ""  
  exit 1
fi
      
echo "  Phenotype file to use: ${phenofile}"  
echo ""




## +++ Call Rscript 

echo "  fetch_pheno.R  ${phenofile}  ${headerfile}  ${field}  ${outfile}"

fetch_pheno.R  ${phenofile}  ${headerfile}  ${field}  ${outfile}  

echo ""





## +++ Finish 

rm -f  ${headerfile1}  ${headerfile2}  ${headerfile3}       

echo ""
if [ -f ${outfile} ]; then
  ls -l ${outfile}
fi
echo ""
 


