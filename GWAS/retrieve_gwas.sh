#!/usr/bin/env bash


# uwemenzel@gmail.com  

 
## === Retrieve tarred gwas project (.tar.gz)  [ compressed before using archive_gwas ]




## +++ Calling:
#
# retrieve_gwas --id ID



 

## +++ Hardcoded settings & and defaults 

setfile=~/archive_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings  
else
  echo ""
  echo "  ERROR (retrieve_gwas.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi





 
## +++ Command line parameters (override the settings in $setfile):

prog=$( basename "$0" )

if [ "$#" -lt 2 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -i|--id <string>               no default"
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

to_test=(ident)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (retrieve_gwas.sh): mandatory variable $var is not defined."
    echo ""
  fi    
done







## +++ Check if the .tar.gz exists:

if [ -s "${ident}.tar.gz" ]; then
  echo ""
  echo "  Going to retrieve the archive ${ident}.tar.gz ..." 
  echo ""
else 
  echo ""
  echo "  ERROR (retrieve_gwas.sh): Archive ${ident}.tar.gz not found."
  echo ""
  exit 1	  
fi






## +++ Check if folder already exists:

if [ -d "${ident}" ]; then
  echo ""
  echo "  WARNING (retrieve_gwas.sh): A folder '${ident}' already exists in the current location." 
  echo "  Delete this folder to renew previous results, use another identifier, or rename the existing folder."
  echo ""
  exit 1	  
fi





## +++ Convert the time string for sbatch command below:

hours=$( expr $minutes / 60 )  
min=$( expr $minutes % 60 )    
if [ "$hours" -eq 0 ]; then
  time=${min}":00"
else
  if [ "${#min}" -eq 1 ]; then min="0"${min}; fi  
  time=${hours}":"${min}":00"    # this is the time for a single chromosome to run
fi  
echo "  Requested runtime for each chromosome: ${time}" | tee -a ${log}
echo "" | tee -a ${log}; echo "" | tee -a ${log}




## +++ Compress

account=$( echo $HOSTNAME | awk 'BEGIN{FS="-"} {print $1}' )
c_ident="RETRIEVE"
logf="${ident}_retrieve.log"

echo "  sbatch -A ${account} -p ${partition} -t ${time}  -J ${c_ident} -o ${logf} -e ${logf}  untar_gwas  --id ${ident}"

jobID=$( sbatch -A ${account} -p ${partition} -t ${time}  -J ${c_ident} -o ${logf} -e ${logf}  untar_gwas  --id ${ident} )  





## +++ Finish

echo ""
echo -n "  "  
date 
echo "" 











