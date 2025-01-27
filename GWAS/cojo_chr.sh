#!/usr/bin/env bash


# uwemenzel@gmail.com  


 

## === LD-pruning using GCTA-COJO , for a single chromosome:
   
# 	https://cnsgenomics.com/software/gcta/#COJO
 





## +++ Calling:

# called by cojo_pheno.sh:

# sbatch --dependency=afterok:${convert_jobid} -A ${account} -p ${partition}  -t ${time}  -J ${c_ident} -o ${prune_log} -e ${prune_log}  \
#         cojo_chr  --id ${ident}  --genoid  ${genoid} --chr ${chrom} --summary ${summary_file} --pval ${pval} --window ${cojo_window} --siglist ${signif_list} --out ${out_prefix}  





 
## +++ Hardcoded settings & defaults  

shopt -s nullglob 

setfile=~/cojo_settings.sh
if [ -s "${setfile}" ];then
  source ${setfile}  # command line paramters overwrite these settings (but not all can be overwritten) 
else
  echo ""
  echo "  ERROR (cojo_chr.sh): Could not find the settings file \"${setfile}\"."
  echo ""
  exit 1  
fi






## +++ Command line parameters:

prog=$( basename "$0" )

if [ "$#" -lt 10 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -i|--id <string>               no default"        
  echo "         -g|--genoid <string>           ${setfile}"
  echo "         -c|--chr <int>                 no default" 
  echo "         -s|--summary <file>            no default"    
  echo "         -p|--pval                      ${setfile}"
  echo "         -w|--window <integer>          ${setfile}"
  echo "         -sl|--siglist <file>           no default"  
  echo "         -o|--out                       no default"                    
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
      -g|--genoid)
          genoid=$2
          shift
          ;;	  
      -c|--chr)
          chrom=$2
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
      -w|--window)
          window=$2
          shift
          ;;
      -sl|--siglist)
          signif_list=$2
          shift
          ;;	  
      -o|--out)
          out_prefix=$2
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

to_test=(ident genoid chrom summary_file pval window signif_list out_prefix )

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (cojo_chr.sh): mandatory variable $var is not defined."
    echo ""
    exit 1
  fi    
done






## +++ Check for correct folder:

folder=$( basename "`pwd`" ) 

if [ "${folder}" != "${ident}" ];then
  echo "" 
  echo "  ERROR (cojo_chr.sh): It seems you are in the wrong location." 
  echo "         Current folder is: ${folder}"
  echo "         Identifier is: ${ident}"
  echo "" 
  exit 1 
fi





## +++ Files  


bad_snps="${out_prefix}.badsnps" 	# output of gcta64, text file   	
freq_bad="${out_prefix}.freq.badsnps"   # output of gcta64, text file   
gcta_log="${out_prefix}.log"       	# log for gcta64 
jma_file="${out_prefix}.jma.cojo"       # output of gcta64, file with genome-wide significant markers     
ldr_file="${out_prefix}.ldr.cojo"  	# output of gcta64, text file
cma_file="${out_prefix}.cma.cojo" 	# output of gcta64, text file






## +++ Check availability of input file

if [ ! -s "${signif_list}" ];then
  echo ""
  echo  "  ERROR (cojo_chr.sh): Could not find file \"${signif_list}\"" 
  echo ""
  exit 1  
fi






## +++ Check chromosome name  

if [[ ! ${chrom} =~ ^[0-9]+$ ]];then    # autosomes only!       
  echo ""
  echo  "  ERROR (cojo_chr.sh): Chromosome name is not valid: " ${chrom} 
  echo  "  			Correct syntax is e.g. --chr 22"
  echo ""
  exit 1 
fi   





## +++ Header:

echo ""
START=$(date +%s) #  1574946757
echo -n "  "
date 
echo "  Job identifier: " ${ident}
echo "  Genotype identifier: ${genoid}"
echo "  Genotype input folder: ${genofolder}" 
echo "  Starting job for chromosome ${chrom}"
echo "  Summary statistics: ${summary_file}" 
echo "  GCTA-COJO p-value: ${pval}"
echo "  GCTA-COJO window: ${window}"
echo "  List of sign. markers loaded: ${signif_list}"
echo "  Output file prefix: ${out_prefix}" 
echo  




# + Check how many signif. markers are on the current chromosome ($chrom)  

chr_signif=$( tail -n +2  ${signif_list} | awk 'BEGIN{FS="\t"} {print $2}' )  
nr_signif_chr=$( printf '%s\n' ${chr_signif[@]} | egrep "^[[:space:]]*${chrom}[[:space:]]*$" | wc -l )  # number of sign. markers for $chrom  


if [ "${nr_signif_chr}" -eq 0 ];then    	# no hits for this chromosome 
  echo ""  
  echo "  No significant marker on chromosome ${chrom} according to the list ${signif_list}." 
  echo "  Cancelling analysis of this chromosome." 
  echo "" 
  echo -n "  "  
  date 
  echo "" 
  exit 0
fi


if [ "${ignore_single_hits}" -eq 1 ];then
  if [ "${nr_signif_chr}" -eq 1 ];then    	# just one hit for this chromosome ==> this hit is independent    
    echo ""   
    echo "  Just one significant marker on chromosome ${chrom} according to the list ${signif_list}." 
    echo "  This marker is independent and being added to the output list \"${jma_file}\" " 
    echo "" 
    # save the marker to ${jma_file} (defined above), in the same format as gcta64 would do: (gcta64 saves to ${ident}_cojo_chr${chrom}.jma.cojo)  
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "Chr" "SNP" "bp" "refA" "freq" "b" "se" "p" "n" "freq_geno" "bJ" "bJ_se" "pJ" "LD_r" > ${jma_file} 
    awk -v c=${chrom} 'BEGIN{FS="\t"}{if($2 == c) printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $2,$1,$3,$4,"NA",$5,$6,$7,"NA","NA","NA","NA","NA","NA"}' ${signif_list} >> ${jma_file} 
    echo -n "  "  
    date 
    echo "" 
    exit 0   
  fi 
fi




geno_prefix="${genofolder}/${genoid}_chr${chrom}"   # /proj/sens2019016/GENOTYPES/BED/MF_chr22

gcta64  --bfile ${geno_prefix}  --thread-num 16 --chr ${chrom} --cojo-file ${summary_file} --cojo-slct --cojo-p ${pval} --cojo-wind ${window} --out ${out_prefix} 






## +++ Finish 

# If no independnet signif. marker have been found, the files jma_file, ldr_file, and cma_file won't exist! 

if [ -s "$jma_file" ]; then   # independent markers have been identified 

  echo "" 
  echo "  Output files:"
  echo "" 
  ls -l ${bad_snps} 2>/dev/null
  ls -l ${freq_bad} 2>/dev/null
  ls -l ${jma_file} 2>/dev/null   
  ls -l ${ldr_file} 2>/dev/null   
  ls -l ${cma_file} 2>/dev/null
  #ls -l ${gcta_log}  
  echo ; echo 

  nr_total=$(  wc -l ${summary_file} | awk '{print $1}' )
  nr_total=$(( ${nr_total} - 1 ))
  echo "  Total number of SNPs in summary file: ${nr_total}" 

  nr_jma=$(  wc -l ${jma_file} | awk '{print $1}' )
  nr_jma=$(( ${nr_jma} - 1 ))
  echo "  Number of independent SNPs: ${nr_jma}" 

  if [ -s "$bad_snps" ]; then
    nr_bad=$(  wc -l ${bad_snps} | awk '{print $1}' )   
    nr_bad=$(( ${nr_bad} - 1 ))
    echo "  Number of \"bad\" SNPs: ${nr_bad}" 
  fi

else

  echo "" 
  echo "  ERROR (cojo_chr): No independent markers identified."   
  # this can actually not happen if everything works fine, because chromosomes without sign. markers have been sorted out above
  
fi



rm -f ${gcta_log}  # batchlog is sufficient 

echo ""
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "  Run time: $DIFF seconds" 
echo "" 
echo -n "  "  
date 
echo "  Done." 
echo "" 






