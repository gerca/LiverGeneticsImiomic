/**************************************************
       SNPs IN ASSOCIATION WITH NAFLD AND CLD
 ***************************************************/
 clear
 use "/home/shafqat/data.dta", replace 
  
 file close _all 
  
  file open output using "/home/results_2.txt", replace write
  
  *write headers
    file write output "snp" _tab "outcome" _tab "N" _tab "Odds Ratio" _tab "95%CI" _tab "P_value" _n
  
  *close file 
  
  **loop
  
  *set trace on 
  *set tracedepth 1 
  foreach var of varlist  NAFLD CLD alcohol_status {
  foreach v2 of varlist rs* {
 	logit `var' `v2' sex age PC1-PC20 array
	noi lincom `v2'
	local lowlim = round(exp(r(lb)), 0.01)
	local upplim = round(exp(r(ub)), 0.01)
	file write output  "`v2'" _tab "`var'" _tab (e(N)) _tab (round(exp(r(estimate)), 0.001)) _tab ("(`lowlim', `upplim')") _tab (r(p)) _n
	}
	
}
file close output
import delimited "/home/results_2.txt", clear  
  
