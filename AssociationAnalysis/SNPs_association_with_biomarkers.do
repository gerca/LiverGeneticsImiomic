/*******************************************************************************
         SNPs in association with various biomarkers
  *******************************************************************************/
  
  clear 
  use "/home/shafqat/Liver_Fat_Genetics/Data/data.dta", clear
  
  *sum liver_fat_c
  
  file close _all 
  
  file open output using "/home/shafqat/results_1.txt", replace write
  
  *write headers
  file write output "snp" _tab "outcome" _tab "N" _tab "beta" _tab "se" _tab "p" _n
  
  *close file 
  
  **loop
  
  set trace on 
  set tracedepth 1 
  foreach var of varlist   bmi_a whr_a height_a lnalkaline_phosphatase lnalanine_aminotransferase lnaspartate_aminotransferase lngamma_glutamyltransferase ///
  lndirect_bilirubin lntotal_bilirubin apolipoprotein_a apolipoprotein_b LPA  lnCRP  glucose HbA1c HDL LDL triglycerides cholesterol ///
  urea calcium_a albumin creatinine phosphate  total_protein  urate Cystatin_C IGF1 SHBG vitamin_D {
  foreach v2 of varlist rs* {
  

 	regress  `var' `v2' sex age PC1-PC20 array
	file write output  "`v2'" _tab "`var'" _tab (e(N)) _tab (_b[`v2']) _tab (_se[`v2']) _tab (2*(ttail(e(df_r), abs(_b[`v2']/_se[`v2'])))) _n
	}
}
file close output
import delimited "/home/shafqat/results_1.txt", clear 
