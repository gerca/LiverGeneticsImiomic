#!/usr/bin/env bash


# uwemenzel@gmail.com  




## === Concatenate results of gcta-cojo pruning   


# called by "cojo_pheno.sh" :
# sbatch -A sens2019016 -p node -t 20:00 -J COLLECT -e collect.test -o collect.test cojo_collect  --id  LIV_MULT5 --phenoname liv10 --cstart 1 --cstop 22 --summary LIV_MULT5_liv10__cojo.ma --out LIV_MULT5_liv10__cojo.jma



 
## +++ Hardcoded settings & defaults  

shopt -s nullglob 

# no other parameters necessary 





## +++ Command line parameters:

prog=$( basename "$0" )

if [ "$#" -lt 12 ]; then
  echo ""
  echo "  Usage: ${prog}"
  echo "         -i|--id  <string>          no default" 
  echo "         -pn|--phenoname <string>   no default"  
  echo "         --cstart <chrom>           no default"
  echo "         --cstopt <chrom>           no default"
  echo "         -s|--summary <file>        no default"   # input       
  echo "         -o|--out <file>            no default"   # output 
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
      -s|--summary)
          summary_file=$2 
          shift
          ;;	  	  
      -o|--out)
          signif_file=$2  
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

to_test=(ident  phenoname cstart cstop summary_file signif_file)

for var in  ${to_test[*]}     
do
  if [ -z ${!var+x} ];then
    echo ""
    echo "  ERROR (cojo_collect.sh): mandatory variable $var is not defined."
    echo ""
  fi    
done





## +++ Header    

echo ""
START=$(date +%s) 
echo -n "  "
date 
echo "  Job identifier: " ${ident}
echo "  Phenotype namn: ${phenoname}"
echo "  Chromosomes:  ${cstart} to ${cstop}"
echo "  Infile (summary statistics): ${summary_file}"  
echo "  Outfile (independent markers): ${signif_file}"   
echo "" 





## +++ Concatenate chromosomal cojo results:

chromosomes=$( seq ${cstart} ${cstop} )

orig="${signif_file}.orig"   	#  a merged (over chromosomes) file based on the original gcta-cojo output		 
echo -n > ${orig}		


for chrom in  ${chromosomes[*]}     
do
  out_prefix="${ident}_${phenoname}_cojo_chr${chrom}"	 
  jma_file="${out_prefix}.jma.cojo"             	  
  if [ -s "${jma_file}" ];then					
    tail -n +2 ${jma_file} >> ${orig}  			
  fi
done





## +++ Reformat for final ouput table (fitting to DT command in gwas_report.Rmd)


printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "ID" "CHR" "POS" "OTHER" "A1" "A1_FREQ" "OBS_CT" "BETA" "SE" "P" > ${signif_file} 
awk 'BEGIN{FS="\t"} {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $2,$1,$3,"-",$4,$5,$9,$11,$12,$13}' ${orig} >> ${signif_file}  # final output 




## +++ Add the "other" allele (by combining the input file with the output file of cojo)  

signif_fixed="${signif_file}.fixed"   	 
module load R_packages/3.6.1  	 
 
cojo_allele   ${summary_file}   ${signif_file}  ${signif_fixed} 
 

if [ -s "${signif_fixed}" ];then  # file exists and is not empty
  nr1=$( wc -l ${signif_file}  | awk '{print $1}' )
  nr2=$( wc -l ${signif_fixed} | awk '{print $1}' )   
  if [ "$nr1" -ne "$nr2" ];then
    echo ""
    echo  "  ERROR (cojo_collect.sh): File \"${signif_file}\" could not be fixed by \"cojo_allele.R\" (unequal nr of rows)." 
    echo ""
    exit 1
  else      
    mv -f ${signif_fixed} ${signif_file}   # overwrite!
  fi 
else 
  echo ""
  echo  "  ERROR (cojo_collect.sh): File \"${signif_file}\" could not be fixed by \"cojo_allele\"." 
  echo ""
  exit 1   
fi





## +++ Finish  

rm -f ${orig}
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "  Run time: $DIFF seconds"
echo "" 
echo "  Table of independent markers: ${signif_file}" 
echo "" 
echo -n "  "  
date 
echo "  Done." 
echo "" 
 








