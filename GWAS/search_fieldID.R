#!/usr/bin/env Rscript


# uwemenzel@gmail.com 



## +++ Find a UKBB field ID by keyword:




## +++ Call:
#
# search_fieldID  <keyword>  [notes] 




## +++ Functions, Libraries 

getpt <- function(searchterm, show.notes = FALSE) {
  if(!is.logical(show.notes)) stop(paste("\n\n getpt : Parameter 'show.notes' must be TRUE or FALSE.\n\n")) 
  scripts = Sys.getenv("SCRIPT_FOLDER")
  fn = paste(scripts, "UKBB_phenotype_descriptions.RData", sep="/") 
  if(!file.exists(fn)) stop(paste("\n\n  ERROR(getpt) : Could not find file '", fn, "'.\n\n"))  
  pt = get(load(fn))
  rowindx = which(grepl(searchterm, pt$Field, ignore.case = TRUE))
  if(identical(rowindx, integer(0))) {
    cat(paste("  No results for searchterm '", searchterm, "'\n"))
    invisible(-1)
  } 
  for(indx in rowindx) { 
    cat    = pt[indx,]$Category
    id     = pt[indx,]$FieldID 
    field  = pt[indx,]$Field 
    if(show.notes) {
      notes  = pt[indx,]$Notes 
      cat(paste("  Category", cat, "\tFieldID :", id, "\tField :", field, "\tNotes :", notes, "\n"))
    } else {
      cat(paste("  Category", cat, "\tFieldID :", id, "\tField :", field, "\n"))    
    }
  }
} 




## +++ Command line parameters   

args = commandArgs(trailingOnly = TRUE)    

if(length(args) < 1) {
  cat("\n")
  cat("  Usage: search_fieldID  <keyword>  [notes]\n")  
  cat("\n")
  quit("no")
}

kword  = args[1]
show.notes = FALSE

if(length(args) > 1) {
  if(args[2] == "notes")  show.notes = TRUE
}




## +++ Search


cat(paste("\n  Searching for keyword '", kword, "'\n\n"))

getpt(kword, show.notes = show.notes)

cat("\n\n")


# uwemenel@gmail.com






