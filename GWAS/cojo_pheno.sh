#!/usr/bin/env bash




# uwemenzel@gmail.com  



## === LD pruning using GCTA-COJO for a single penotype ===    
   



## +++ Calling: 

# called by run_cojo.sh
#  
#    cojo_pheno  --id LIV6   --phenoname vox1_exp  --pval 5.0e-8  --window 5000  --part node  --minutes 120 






## +++ Hardcoded settings & defaults 

shopt -s nullglob 

setfile=~/cojo_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings 
else
  echo ""
  echo "  ERROR (cojo_pheno.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi





 
## +++ Programs 
 
prog=$( which gcta64 )   
exit_code=$?  
if [ ${exit_code} -ne 0 ]
then
  echo "" 
  echo "  ERROR (cojo_pheno.sh): Did not find the gcta64 program." 
  echo ""
fi 





## +++ Command line parameters:   

if [ "$#" -lt 4 ]; then
  prog=$( basename "$0" )
  echo ""
  echo "  Usage: ${prog}"
  echo "         -i|--id <string>               no default"
  echo "         -pn|--phenoname <string>       no default"
  echo "         -p|--pval <real>               ${setfile}"
  echo "         -w|--window <integer>          ${setfile}" 
  echo "         -m|--minutes <int>             ${setfile}"
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
      *)
          echo ""
	  echo "  Invalid argument: $1"
	  echo ""
	  exit 1
          ;;
  esac
  shift
done







## +++ Files

paramfile="${ident}_gwas_params.txt" 				
signif_file="${ident}_${phenoname}_cojo.jma"   			 
log="${ident}_${phenoname}_cojo.log"   				  
convert_log="${ident}_${phenoname}_cojo_convert.log"  				  		 
collect_log="${ident}_${phenoname}_cojo_collect.log"		   
clean_log="${ident}_${phenoname}_cojo_clean.log"      		  
summary_file="${ident}_${phenoname}_cojo.ma"   			
signif_list="${ident}_${phenoname}_gwas_signif.txt"  		






## +++ Read remaining parameters from the param files (created in "run_gwas.sh"):

if [ ! -s "$paramfile" ]; then
  echo ""
  echo "  ERROR (cojo_pheno.sh): Missing parameter file ${paramfile}"
  echo ""
  exit 1
fi

# may include cojo results for multiple phenotypes
# may include clump results for multiple phenotypes 

genoid=$( awk '{if($1 == "genotype_id") print $2}' ${paramfile} )  
cstart=$( awk '{if($1 == "cstart") print $2}' ${paramfile} )  	
cstop=$( awk '{if($1 == "cstop") print $2}' ${paramfile} )  	

# remove possibly existing cojo-entry for this phenotype: 
rnum=$(( 1 + RANDOM%10000 ))   # avoid interference with jobs running parallel
awk -v pheno=$phenoname '{if(!($1 == "cojo_out" && $2 == pheno)) {print $0}}' ${paramfile} > temp_${phenoname}_${rnum}.txt 
mv temp_${phenoname}_${rnum}.txt ${paramfile}  
echo "cojo_out ${phenoname} ${signif_file}" >> ${paramfile} 	# replace by current entry 





## +++ Check if the variables are defined  

to_test=(ident phenoname pval window partition minutes genoid cstart cstop minspace)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (cojo_pheno.sh): mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done





## +++ Check folder:

folder=$( basename "`pwd`" ) 

if [ "${folder}" != "${ident}" ];then
  echo "" 
  echo "  ERROR (cojo_pheno): It seems you are in the wrong location." 
  echo "         Current folder is: ${folder}"
  echo "         Identifier is: ${ident}"
  echo "" 
  exit 1 
fi






## +++ Check if $phenoname is valid (user input)

pname=$( awk '{if($1 == "phenoname") print $2}' ${paramfile} )      
pname=$( echo $pname | tr -s ',' '\t' )  			   
parray=($pname)
# echo " Number of elements in parray: ${#parray[*]}"   		#  10 ok  
nr_hits=$( printf '%s\n' ${parray[@]} | egrep "^[[:space:]]*${phenoname}[[:space:]]*$" | wc -l )  # should exactly be 1
if [ "${nr_hits}" -ne 1 ];then
  echo ""
  echo "  ERROR (cojo_pheno.sh): You propably picked a wrong phenotype name."
  echo "  The word \"${phenoname}\" is not included as a phenoname entry in \"${paramfile}\""
  echo ""
  exit 1
fi





## +++ Check chromosomes:

if [[ ! ${cstart} =~ ^[0-9]+$ ]];then
  echo ""
  echo  "  ERROR (cojo_pheno.sh): Start chromosome is not valid: " ${cstart} 
  echo  "  			  Correct syntax is e.g. --chrom 1-16"
  echo ""
  exit 1 
fi   

if [[ ! ${cstop} =~ ^[0-9]+$ ]];then
  echo ""
  echo  "  ERROR (cojo_pheno.sh): Stop chromosome is not valid: " ${cstop} 
  echo  "  			  Correct syntax is e.g. --chrom 1-16"
  echo ""
  exit 1 
fi   

chromosomes=$( seq ${cstart} ${cstop} )






## +++ Check available disk space:

space=$( df -k . | tail -1 | awk '{print $4}' )  # kb  22430291840    
spac1=$( df -h . | tail -1 | awk '{print $4}' )  # human readable  21T 
if [ ${space} -lt ${minspace} ]; then	
    echo "" 
    echo "  Less than ${minspace} disk space available, consider using a different location." 
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
  time=${hours}":"${min}":00"  # requested runtime for a single chromosome
fi  





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




## +++ Header:

account=$( echo $HOSTNAME | awk 'BEGIN{FS="-"} {print $1}' ) 	# sens2019016
echo ""  > ${log}
echo ""   | tee -a ${log}
START=$(date +%s)      #  1574946757
echo -n "  "  | tee -a ${log}
date | tee -a ${log}
echo "  Account: ${account}" | tee -a ${log}
echo -n "  Operated by: " | tee -a ${log} 
whoami | tee -a ${log} 
echo -n "  Current working folder: "  | tee -a ${log}
pwd  | tee -a ${log}
echo "  Available disk space in this path: ${spac1}"  | tee -a ${log}
echo "  Job identifier:  ${ident}" | tee -a ${log}
echo "  Phenotype namn: ${phenoname}" | tee -a ${log}
echo "  GCTA-COJO window: ${window}" | tee -a ${log}
echo "  GCTA-COJO p-value: ${pval}" | tee -a ${log}
echo "  Genotype identifier: ${genoid}" | tee -a ${log}
echo "  Running on chromosomes ${cstart} to ${cstop}" | tee -a ${log}
echo "  Master logfile: ${log}" | tee -a ${log}
echo "" | tee -a ${log}
echo "  Requested partition: ${partition}" | tee -a ${log}
echo "  Requested runtime per chromosome: ${minutes} minutes." | tee -a ${log}
echo "  Requested runtime per chromosome: ${time}" | tee -a ${log}
echo "" | tee -a ${log}  







## +++ Create COJO input file  

c_ident="CONVERT"
# cojo_convert_time="20:00" 			# moved to ~/cojo_settings.sh   time requested for "cojo_convert.sh" 

echo "sbatch -A ${account} -p ${partition}  -t ${cojo_convert_time}  -J ${c_ident} -o ${convert_log} -e ${convert_log} \  " | tee -a ${log}
echo "       cojo_convert  --id ${ident}  --phenoname ${phenoname} --summary ${summary_file} --pval ${pval}  --siglist ${signif_list}"  | tee -a ${log} 

convert_jobid=$( sbatch -A ${account} -p ${partition}  -t ${cojo_convert_time}  -J ${c_ident} -o ${convert_log} -e ${convert_log} \
       cojo_convert  --id ${ident}  --phenoname ${phenoname} --summary ${summary_file} --pval ${pval}  --siglist ${signif_list} )

convert_jobid=$( echo $convert_jobid | awk '{print $NF}' )
echo "    JobID for cojo-convert : ${convert_jobid}" | tee -a ${log}   

echo "" | tee -a ${log}




 
## +++ Run through chromosomes having significant hits with cojo:


let i=0
for chrom in  ${chromosomes[*]}     
do

  let i++
  
  prune_log="${ident}_${phenoname}_cojo_chrom${chrom}.log"	  
  c_ident="COJO-${chrom}"  
  out_prefix="${ident}_${phenoname}_cojo_chr${chrom}"  		  

  echo "" | tee -a ${log}
  echo "sbatch --dependency=afterok:${convert_jobid} -A ${account} -p ${partition}  -t ${time}  -J ${c_ident} -o ${prune_log} -e ${prune_log}  \  "  | tee -a ${log} 
  echo "     cojo_chr  --id ${ident}  --genoid  ${genoid} --chr ${chrom} --pval ${pval} --window ${window} --summary ${summary_file} --siglist ${signif_list} --out ${out_prefix}" | tee -a ${log} 
  
  jobid=$( sbatch --dependency=afterok:${convert_jobid} -A ${account} -p ${partition} -C mem512GB -t ${time}  -J ${c_ident} -o ${prune_log} -e ${prune_log}  \
          cojo_chr  --id ${ident}  --genoid  ${genoid} --chr ${chrom} --pval ${pval} --window ${window} --summary ${summary_file} --siglist ${signif_list} --out ${out_prefix} ) 

  jobid=$( echo $jobid | awk '{print $NF}' ) 
  echo "    JobID for chromosome ${chrom} : ${jobid}" | tee -a ${log}  
  
  if [ "$i" -eq 1 ]; then
    liste="${jobid}"
  else
    liste="${liste}:${jobid}"  
  fi
 
done   
 
echo "" | tee -a ${log} 
echo "" | tee -a ${log} 





## +++ Concatenate output files for individual chromosomes:  

c_ident="COLLECT"			 

echo "sbatch --dependency=afterok:${liste} -A ${account} -p ${partition}  -t ${cojo_collect_time}  -J ${c_ident} -o ${collect_log} -e ${collect_log} \ " | tee -a ${log} 
echo "       cojo_collect  --id ${ident} --phenoname ${phenoname} --cstart ${cstart}  --cstop ${cstop} --summary ${summary_file} --out ${signif_file} "  | tee -a ${log} 

collect_jobid=$( sbatch --dependency=afterok:${liste} -A ${account} -p ${partition}  -t ${cojo_collect_time}  -J ${c_ident} -o ${collect_log} -e ${collect_log} \
       cojo_collect  --id ${ident} --phenoname ${phenoname} --cstart ${cstart}  --cstop ${cstop} --summary ${summary_file} --out ${signif_file} )  
collect_jobid=$( echo $collect_jobid | awk '{print $NF}' )
echo "    JobID for cojo-collect : ${collect_jobid}" | tee -a ${log} 
echo "" | tee -a ${log} 





## +++ Clean 

c_ident="CLEAN"							  

echo "sbatch --dependency=afterok:${collect_jobid} -A ${account} -p ${partition}  -t ${cojo_clean_time}  -J ${c_ident} -o ${clean_log} -e ${clean_log}  \ " | tee -a ${log} 
echo "       cojo_clean  --id ${ident} --phenoname ${phenoname} --cstart ${cstart}  --cstop ${cstop}" | tee -a ${log} 

clean_jobid=$( sbatch --dependency=afterok:${collect_jobid} -A ${account} -p ${partition}  -t ${cojo_clean_time}  -J ${c_ident} -o ${clean_log} -e ${clean_log}  \
	cojo_clean  --id ${ident} --phenoname ${phenoname} --cstart ${cstart}  --cstop ${cstop} )   
clean_jobid=$( echo $clean_jobid | awk '{print $NF}' )
echo "    JobID for cojo-clean : ${clean_jobid}" | tee -a ${log}
echo "" | tee -a ${log} 





## +++ Finish  

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "  Run time: $DIFF seconds"| tee -a ${log}
echo "" | tee -a ${log}
echo "  Table of independent markers being created: ${signif_file}" | tee -a ${log}   # this file is not available at this point (job still in queue)! 
echo "" | tee -a ${log} 
echo -n "  "  | tee -a ${log}
date | tee -a ${log}
echo "  Done." | tee -a ${log}
echo "" | tee -a ${log}
 
 


 
 
 
