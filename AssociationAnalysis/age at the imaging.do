/********************************************************************************
            DATA FILE FOR THE LIVER FAT GENETICS PROJECT
	    
This script is prepared for the age at the imaging  variable
*********************************************************************************/

clear
import delimited /proj/sens2019016/20231221_image_file_visit/imaging_date.csv, delimiter(comma) bindquote(strict) varnames(nonames) case(preserve) asdouble clear
rename v1 iid
rename v2 baseline
drop v3 v5 
rename v4 first_visit
drop if iid=="eid"
*drop if first_visit == ""
destring iid, replace 
save "/home/shafqat/imaging_date.dta", replace

clear 
use "/home/shafqat/pheno.dta", replace
merge 1:1 iid using "/home/shafqat/imaging_date.dta"
drop if liver_fat==.
drop if first_visit == ""
drop _merge
gen difference_date=((date(first_visit,"YMD")-date(baseline,"YMD")))/365.24
gen age_at_follow_up= age + difference_date
save "/home/shafqat/liver_fat_imaging_date.dta", replace






