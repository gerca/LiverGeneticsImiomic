#!/usr/bin/env bash


# uwemenzel@gmail.com  



## === Convert plink2 gwas output files (*.glm.linear.gz) to GCTA-COJO input file (*.ma)   
   
# 	https://cnsgenomics.com/software/gcta/#COJO






## +++ Calling
#
# called by "run_cojo.sh" :
#
# sbatch -A ${account} -p ${partition}  -t ${ctime}  -J ${c_ident} -o ${sbatch_log} -e ${sbatch_log} \
#        cojo_convert.sh  --id ${ident}  --phenoname ${phenoname}  --summary ${summary_file} --pval ${pval}  --siglist ${signif_list}
#
# sbatch -A sens2019016 -p node  -t 20:00  -J CONVERT -o cojo_convert.log -e cojo_convert.log \
#        cojo_convert.sh  --id LIV_MULT4  --phenoname liv1  --summary LIV_MULT4_liv1_cojo.ma --pval 5e-8  --siglist LIV_MULT4_liv1_gwas_signif.txt




 
## +++ Hardcoded settings & defaults  

shopt -s nullglob 

setfile=~/cojo_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings  
else
  echo ""
  echo "  ERROR (cojo_convert.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi





## +++ Get command line parameters:

prog=$( basename "$0" )

if [ "$#" -lt 8 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -i|--id <string>               no default"  
  echo "         -pn|--phenoname <string>       no default"        
  echo "         -s|--summary <file>            no default"   # output file  (.ma)
  echo "         -p|--pval <real>               ${setfile}"
  echo "         -sl|--siglist <file>           no default"   # output file (signif. markers across the genome)  
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
      -s|--summary)
          summary_file=$2
          shift
          ;;
      -p|--pval)
          pval=$2
          shift
          ;;
      -sl|--siglist)
          signif_list=$2
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

to_test=(ident phenoname  summary_file  signif_list pval  partition)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (cojo_convert.sh): Mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done






## +++ Header:      

echo ""
START=$(date +%s) 
echo -n "  "
date 
echo "  Job identifier: ${ident}"
echo "  Phenotype namn: ${phenoname}"
echo "  Summary statistics to create: ${summary_file}" 
echo "  p-value threshold (for creating list of signif. markers): ${pval}" 
echo "  List with significant markers (whole genome): ${signif_list}  (to create)"
echo "" 






## +++  Regression output files:   

regress_out=(${ident}_gwas_chr*${phenoname}.glm.linear)    

nr_regress=${#regress_out[*]}    

if [ "${nr_regress}" -lt 1 ];then
  echo ""
  echo "  ERROR (cojo_convert.sh): No regression output files found."
  echo ""
  exit 1
fi
 
echo "" 
echo "  GWAS results in this folder:" 
echo "" 

let i=0
for glm in  ${regress_out[*]} 
do
  let i++
  ls -l $glm 
done
echo "" 






## +++ Concatenate the summary statistics of all chromosomes: 

sumstat_all_chrom="${ident}_${phenoname}.glm.linear"  # temporary

echo -n "  Merging gwas files ..."  	# following file name convention from plink2 --glm (gwas_chr.sh) 
tail -n +2 -q ${ident}_gwas_chr*${phenoname}.glm.linear > ${sumstat_all_chrom}     
echo  "  Done."  





## +++ Convert plink2 gwas output file to the format requested by gcta:  see https://cnsgenomics.com/software/gcta/#COJO  

echo -n "  Reformatting gwas files for GCTA-COJO ..."  
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "ID" "A1" "OTHER" "A1_FREQ" "BETA" "SE" "P" "OBS_CT" > ${summary_file} # ${ident}_cojo.ma (run_cojo.sh)
 
awk 'BEGIN{FS="\t"} {if($6 == $5) {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $3, $6, $4, $7, $9, $10, $11, $8} \
                             else {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $3, $6, $5, $7, $9, $10, $11, $8}}' ${sumstat_all_chrom} >> ${summary_file} 
echo "  Done." 





# + Check the header (this is critical!)    

header=$( head -1 ${summary_file} ) 
echo "" 

arr=($header)
nr_cols=${#arr[@]}

if [ "$nr_cols" -ne 8 ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Need 8 columns in the cojo input file ( \"${summary_file}\" ), not ${nr_cols}." 
  echo "" 
  exit 1
fi

if [ ${arr[0]} != "ID" ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Wrong column 1 in summary statstistics file: ${arr[0]}" 
  echo "" 
  exit 1
fi

if [ ${arr[1]} != "A1" ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Wrong column 2 in summary statstistics file: ${arr[1]}" 
  echo "" 
  exit 1
fi

if [ ${arr[2]} != "OTHER" ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Wrong column 3 in summary statstistics file: ${arr[2]}"  
  echo "" 
  exit 1
fi

if [ ${arr[3]} != "A1_FREQ" ];then
  echo ""
  echo "  ERROR (cojo_convert.sh): Wrong column 4 in summary statstistics file: ${arr[3]}" 
  echo ""
  exit 1
fi

if [ ${arr[4]} != "BETA" ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Wrong column 5 in summary statstistics file: ${arr[4]}" 
  echo "" 
  exit 1
fi

if [ ${arr[5]} != "SE" ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Wrong column 6 in summary statstistics file: ${arr[5]}" 
  echo ""
  exit 1
fi

if [ ${arr[6]} != "P" ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Wrong column 7 in summary statstistics file: ${arr[6]}"  
  echo "" 
  exit 1
fi

if [ ${arr[7]} != "OBS_CT" ];then
  echo "" 
  echo "  ERROR (cojo_convert.sh): Wrong column 8 in summary statstistics file: ${arr[7]}"  
  echo "" 
  exit 1
fi







## +++ Create a list including chromosomes with significant markers (to exclude those chromosomes with no signif. markers from analysis in "cojo_chr.sh")

echo -n "  Extracting significant markers ..."
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "ID" "CHR" "POS" "A1" "BETA" "SE" "P" > ${signif_list}
awk -v pval=${pval} 'BEGIN{FS="\t"}{if($NF <= pval) printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $3,$1,$2,$6,$9,$10,$11}' ${sumstat_all_chrom} >> ${signif_list}
nr_signif=$( wc -l ${signif_list} | awk '{print $1}' )
nr_signif=$(( ${nr_signif} - 1 ))
echo "  Done (${nr_signif} markers across the genome, saved to ${signif_list})." 




## +++ Finish 

rm -f ${sumstat_all_chrom}   		
echo "" 
echo "  Output file: ${summary_file}"   
echo ""
echo -n "  "  
date 
echo ""
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "  Run time: $DIFF seconds"
echo "" 
echo "  Done." 
echo "" 






