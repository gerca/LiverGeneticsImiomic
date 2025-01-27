#!/usr/bin/env bash
  

# uwemenzel@gmail.com  



## === LD pruning using GCTA-COJO for single, multiple (or all) phenonames of the current gwas run ===    
   




## +++ Calling: 

#  run_cojo  --id LIV_MULT2      				   
#
#  run_cojo  --id LIV_MULT2  --phenoname liv1,liv2,liv3,liv4,liv5   
#
#  with all command line parameters:
#
#  run_cojo  --id LIV_MULT2  --phenoname liv1,liv2,liv3,liv4,liv5 --pval 5e-8  --window 5000  --minutes 100 --ask y







## +++ Hardcoded settings & and defaults 

shopt -s nullglob 

setfile=~/cojo_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings (but not all can be overwritten) 
else
  echo ""
  echo "  ERROR (run_cojo.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi





 
## +++ Programs 
 
prog=$( which gcta64 )   
exit_code=$?  
if [ ${exit_code} -ne 0 ]
then
  echo "" 
  echo "  ERROR (run_cojo.sh): Did not find script gcta64." 
  echo ""
fi 






## +++ Command line parameters:   

if [ "$#" -lt 2 ]; then
  prog=$( basename "$0" )
  echo ""
  echo "  Usage: ${prog}"
  echo "         -i|--id <string>               no default"
  echo "         -pn|--phenoname <string>       defaults to all gwas results"
  echo "         -p|--pval <real>               ${setfile}"
  echo "         -w|--window <integer>          ${setfile}" 
  echo "         -m|--minutes <int>             ${setfile}"
  echo "         --ask <y|n>                    ${setfile}"      
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
      -p|--pval)
          pval=$2
          shift
          ;;	  
      -w|--window)
          window=$2
          shift
          ;;	  
      -m|--minutes)
          minutes=$2
          shift
          ;;
      --ask)
          ask=$2
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





## +++ Read remaining parameters from the param files (created in "run_gwas.sh"):
 
paramfile="${ident}_gwas_params.txt"

if [ ! -s "$paramfile" ]; then
  echo ""
  echo "  ERROR (run_cojo.sh): Missing parameter file ${paramfile}"
  echo ""
  exit 1
fi

genoid=$( awk '{if($1 == "genotype_id") print $2}' ${paramfile} )  
cstart=$( awk '{if($1 == "cstart") print $2}' ${paramfile} )  	
cstop=$( awk '{if($1 == "cstop") print $2}' ${paramfile} )  	







## +++ Check if the variables are defined  

to_test=(ident pval window partition minutes ask genoid cstart cstop minspace  sleep_between_pheno)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (run_cojo.sh): mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done





## +++ Check folder:

folder=$( basename "`pwd`" ) 

if [ "${folder}" != "${ident}" ];then
  echo "" 
  echo "  ERROR (run_cojo.sh): It seems you are in the wrong location." 
  echo "         Current folder is: ${folder}"
  echo "         Identifier is: ${ident}"
  echo "" 
  exit 1 
fi





## +++ Check chromosomes:    

if [[ ! ${cstart} =~ ^[0-9]+$ ]];then
  echo ""
  echo  "  ERROR (run_cojo.sh): Start chromosome is not valid: " ${cstart} 
  echo  "  			Correct syntax is e.g. --chrom 1-16"
  echo ""
  exit 1 
fi   

if [[ ! ${cstop} =~ ^[0-9]+$ ]];then
  echo ""
  echo  "  ERROR (run_cojo.sh): Stop chromosome is not valid: " ${cstop} 
  echo  "  			Correct syntax is e.g. --chrom 1-16"
  echo ""
  exit 1 
fi   





## +++ Check available disk space:

space=$( df -k . | tail -1 | awk '{print $4}' )  
spac1=$( df -h . | tail -1 | awk '{print $4}' )  
if [ ${space} -lt ${minspace} ]; then   
    echo "" 
    echo "  Less than ${minspace} disk space available, consider using a different location." 
    echo "" 
    exit 1 
fi 





## +++ If not provided on command line, read phenonames from the parameter file (created in "run_gwas.sh") :
#    in that case, all phenonames run through gwas will be considered (high workload!) 
 
if [ -z ${phenoname+x} ];then

  echo ""
  echo "  No phenotype names invoked on command line, reading from parameter file."
   
  if [ ! -s "$paramfile" ]; then
    echo ""
    echo "  ERROR (run_cojo.sh): Missing parameter file ${paramfile}"
    echo ""
    exit 1
  fi

  phenoname=$( grep phenoname ${paramfile} | awk '{print $2}' )  

fi   

pname=$( echo $phenoname | tr -s ',' '\t' )  
phenoarray=($pname)
nr_pnames=${#phenoarray[*]} 
echo  
echo "  Number of phenotype names: ${nr_pnames}" 





## +++ Check if $phenoname is valid (user input)


paramfile_names=$( awk '{if($1 == "phenoname") print $2}' ${paramfile} ) 
paramfile_names=$( echo $paramfile_names | tr -s ',' '\t' )              
paramfile_array=($paramfile_names)					 


for pheno in  ${phenoarray[*]} 
do
  nr_hits=$( printf '%s\n' ${paramfile_array[@]} | egrep "^[[:space:]]*${pheno}[[:space:]]*$" | wc -l )
  if [ "${nr_hits}" -ne 1 ];then
    echo "" 
    echo "  ERROR (run_cojo.sh): Invoked phenotype name \"${pheno}\" is not valid (check in \"${paramfile}\")"
    echo ""
    exit 1
  fi
done




## +++ Check if alternative genotype files were provided

if [ -z ${alt_genoid} ];then    # -z does NOT exist 
  echo ""
  echo "  No alternative genotype ID defined. Using \"${genoid}\" read from \"${paramfile}\""
  echo ""
else
  echo ""
  genoid=${alt_genoid}
  echo "  An alternative genotype ID was defined: \"${genoid}\" read from \"${paramfile}\""
  echo ""
fi    



## +++ Header     

START=$(date +%s)  
echo "  Job identifier:  ${ident}" 
echo "  Genotype input folder: ${genofolder}" 
echo "  Genotype identifier: ${genoid}"  
echo "  Phenotype name(s): ${phenoname}" 
echo "  GCTA-COJO window (kB): ${window}" 
echo "  GCTA-COJO p-value: ${pval}" 
echo "  Running on chromosomes ${cstart} to ${cstop}" 
echo "  Requested partition: ${partition}" 
echo "  Requested runtime per chromosome: ${minutes} minutes." 
echo "" 








## +++ Check availability of genotype files:

for chrom in  ${chromosomes[*]}     
do

  pgen_prefix=${genofolder}"/"${genoid}"_chr"$chrom       

  psam=${pgen_prefix}".psam"	
  pvar=${pgen_prefix}".pvar"	 
  pgen=${pgen_prefix}".pgen" 	   

  if [ ! -f ${psam} ]; then
    echo "" 
    echo "  ERROR (run_cojo.sh): Input file '${psam}' not found." 
    echo "" 
    exit 1 
  fi  

  if [ ! -f ${pvar} ]; then
    echo "" 
    echo "  ERROR (run_cojo.sh): Input file '${pvar}' not found." 
    echo "" 
    exit 1 
  fi  

  if [ ! -f ${pgen} ]; then
    echo "" 
    echo "  ERROR (run_cojo.sh): Input file '${pgen}' not found." 
    echo "" 
    exit 1 
  fi    

done   

echo "  All required genotype files (.pgen, .pvar, .psam) are available."  





## +++ Let the user confirm the choice of parameters if $ask = "y"

if [[ "$ask" =~ ^y+ ]];then  
  nr_nodes=$(( 25*${nr_pnames} ))  # or cores, depending of $partition
  echo ""
  echo "  Pruning with ${nr_pnames} phenotype(s) will start ${nr_nodes} ${partition}s in parallel."  
  echo ""
  read -p "  Do you want to proceed ? (y/n):" -n 1 -r  
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then 
    echo "  Starting COJO"; echo
  else
    echo; echo "  Bye."; echo
    exit 0
  fi
fi





## +++ Start a cojo for each phenoname:

sleeptime="${sleep_between_pheno}m"

for pheno in  ${phenoarray[*]} 
do
  echo "  cojo_pheno.sh --id ${ident} --phenoname  ${pheno} --pval ${pval}  --window ${window} --minutes ${minutes}"
  cojo_pheno.sh --id ${ident} --phenoname  ${pheno} --pval ${pval}  --window ${window} --minutes ${minutes}
  
  sleep ${sleeptime}
done






## +++ Finish  

END=$(date +%s)
DIFF=$(( $END - $START ))
echo ""
echo "  Run time: $DIFF seconds (but we have sent a lot of jobs to the batch)"
echo "" 
echo -n "  " 
date 
echo "  Done." 
echo "" 
 
 
















