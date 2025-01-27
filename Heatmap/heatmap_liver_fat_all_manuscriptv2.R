# Clear the workspace
rm(list=ls())

# Set the seed for reproducibility
set.seed(123)

# Load necessary libraries
library(rio)           # For importing/exporting data
library(reshape2)      # For reshaping data (e.g., `dcast`)
library(ComplexHeatmap) # For creating advanced heatmaps
library(circlize)      # Color mapping for heatmaps
library(RColorBrewer)  # Color palettes
library(viridis)       # Perceptually uniform color scales

# Load the dataset
res=import("//argos.rudbeck.uu.se/MyGroups$/Gold/MolEpi/Personal folders and scripts/ssayols/Uppsala_university/Shafqat/snp_phenotype_biomarkers_alcohol_NALD_CLD_heatmap_file_1_13DEC2023.txt")

# Add the Gene information to SNP names
res$snp = paste(res$snp, " (", res$Gene, ")", sep="")

# Select relevant columns
res = res[,c("snp", "outcome", "beta", "p", "SNP_associated_trait")]

# Standardize names of outcomes
res$outcome=ifelse(res$outcome%in%"lnCRP","CRP",
                   ifelse(res$outcome%in%"DBIL",
                          "Direct bilirubin",
                          ifelse(res$outcome%in%"TBIL",
                                 "Total Bilirubin",
                                 ifelse(res$outcome%in%"ALB",
                                        "Albumin",
                                        ifelse(res$outcome%in%"BUN",
                                               "Urea",
                                               ifelse(res$outcome%in%"CA",
                                                      "Calcium",
                                                      ifelse(res$outcome%in%"PHOS",
                                                             "Phosphate",
                                                             ifelse(res$outcome%in%"CHOL",
                                                                    "Cholesterol",
                                                                    ifelse(res$outcome%in%"CREA",
                                                                           "Creatinine",
                                                                           ifelse(res$outcome%in%"TP",
                                                                                  "Total Protein",
                                                                                  ifelse(res$outcome%in%"UA",
                                                                                         "Urate",
                                                                                         ifelse(res$outcome%in%"TG",
                                                                                                "Triglycerides",
                                                                                                ifelse(res$outcome%in%"HDL",
                                                                                                       "HDL cholesterol",
                                                                                                       ifelse(res$outcome%in%"LDL",
                                                                                                              "LDL cholesterol",
                                                                                                              ifelse(res$outcome%in%"GLU",
                                                                                                                     "Glucose",
                                                                                                                     ifelse(res$outcome%in%"LPA",
                                                                                                                            "Lp(a)",
                                                                                                                            ifelse(res$outcome%in%"HBA1C",
                                                                                                                                   "HbA1c",
                                                                                                                                   ifelse(res$outcome%in%"CYS",
                                                                                                                                          "Cystain C",
                                                                                                                                          ifelse(res$outcome%in%"VITD",
                                                                                                                                                 "Vitamin D",
                                                                                                                                                 ifelse(res$outcome%in%"NAFLD","MASLD",res$outcome))))))))))))))))))))




# Handle special cases for SNPs
res$snp = ifelse(res$snp %in% "Liver fat ()", "Liver fat",
            ifelse(res$snp %in% "Liver volume ()", "Liver volume", res$snp))

# Filter out unwanted SNPs
res = res[which(res$snp %in% grep("lcohol", res$snp, value=T) == F),]
res = res[which(res$snp %in% c("rs1490384 ()", "rs75460349 ()") == F),]
res = res[which(res$snp %in% c("Liver fat", "Liver volume") == F),]

# Transform beta values for specific outcomes
res[which(res$outcome %in% c("MASLD", "CLD") == T), "beta"] = exp(res[which(res$outcome %in% c("MASLD", "CLD") == T), "beta"])

# Create annotations for SNP traits
anot = unique(res[,c("snp", "SNP_associated_trait")])
anot[which(anot$snp %in% "Liver fat"), "SNP_associated_trait"] = "Liver Fat"
anot[which(anot$snp %in% "Liver volume"), "SNP_associated_trait"] = "Liver Volume"

# Standardize annotations
anot$SNP_associated_trait = ifelse(anot$SNP_associated_trait == "Liver Fat", "Liver fat",
                               ifelse(anot$SNP_associated_trait == "Liver Volume", "Liver volume", NA))

# Function to prepare data for heatmaps
prepare.plot = function(res) {
  res.plot.model = dcast(res[which(res$outcome %in% c("MASLD", "CLD") == F),], snp ~ outcome, value.var="beta")
  res.plot.model.q = dcast(res[which(res$outcome %in% c("MASLD", "CLD") == F),], snp ~ outcome, value.var="p")

  res.plot.model.q[,2:ncol(res.plot.model.q)] = apply(res.plot.model.q[,2:ncol(res.plot.model.q)], 2, as.numeric)

  rownames(res.plot.model) = rownames(res.plot.model.q) = res.plot.model[,1]
  res.plot.model = res.plot.model[,2:ncol(res.plot.model)]
  res.plot.model.q = res.plot.model.q[,2:ncol(res.plot.model.q)]
  res.plot.model = as.matrix(res.plot.model)
  res.plot.model.q = as.matrix(res.plot.model.q)

  res.plot.model.q = ifelse(res.plot.model.q < 0.05 / (length(unique(res$snp)) * length(unique(res$outcome))), "*", "")
  res.plot.model[is.na(res.plot.model)==T] = 0
  res.plot.model.q[is.na(res.plot.model.q)==T] = ""

res.plot.model2=dcast(res[which(res$outcome%in%c("MASLD","CLD")==T),],snp~outcome,value.var = "beta")
  res.plot.model.q2=dcast(res[which(res$outcome%in%c("MASLD","CLD")==T),], snp ~ outcome,value.var="p")
  res.plot.model.q2[,2:ncol(res.plot.model.q2)]=apply(res.plot.model.q2[,2:ncol(res.plot.model.q2)],2,as.numeric)
  
  
  rownames(res.plot.model2)=rownames(res.plot.model.q2)=res.plot.model2[,1]
  res.plot.model2=res.plot.model2[,2:ncol(res.plot.model2)]
  res.plot.model.q2=res.plot.model.q2[,2:ncol(res.plot.model.q2)]
  
  res.plot.model2 <- as.matrix(res.plot.model2)
  res.plot.model.q2 <- as.matrix(res.plot.model.q2)
  
  res.plot.model.q2 <- ifelse(res.plot.model.q2<0.05/(length(unique(res$snp))*length(unique(res$outcome))),"*","")
  
  res.plot.model2[which(is.na(res.plot.model2)==T)]=1
  res.plot.model.q2[which(is.na(res.plot.model.q2)==T)]=""
  
  
  rwname=c("rs429358 (APOE)","rs188247550 (TM6SF2)","rs8107974 (TM6SF2)","rs738408 (PNPLA3)","rs2304128 (LPAR2)","rs7896518 (REEP3)","rs10881959 (TNKS2)","rs6858148 (ADH4)","rs853966 (CENPW)","rs1260326 (GCKR)","rs79287178 (TNFSF10)","rs139974673 (PDIA3)","rs193084249 (ARID1A)","rs4240624 (PPP1R3B)")
  
  
  return(list(res.plot.model=res.plot.model[rwname,],
              res.plot.model.q=res.plot.model.q[rwname,],
              res.plot.model2=res.plot.model2[rwname,],
              res.plot.model.q2=res.plot.model.q2[rwname,]))

}

# Prepare data for heatmap plotting
res.plot = prepare.plot(res)

# Create row annotations
x.anot = anot$SNP_associated_trait
names(x.anot) = anot$snp
row_ha = rowAnnotation("Trait" = x.anot[rownames(res.plot$res.plot.model)], 
                       col = list(Trait = c("Liver fat" = "#E6AB02", "Liver volume" = "#66A61E")),
                       annotation_name_rot = 45)

############################# 
# Create heatmaps for continuous and discrete outcomes (h1 and h2)
############################# 

h1 = Heatmap(res.plot$res.plot.model, 
             column_names_rot =  45,
             column_names_side = "bottom",
             clustering_method_columns="ward.D2",
             cluster_columns =T,
             cluster_rows = F,
             name="Beta",
             column_names_gp = gpar(fontsize =10),
             row_names_gp = gpar(fontsize = 10),
             left_annotation = row_ha ,
             heatmap_legend_param = list(
               at=c(-0.2,0,0.2,0.4),
               labels=c("-0.2","0","0.2","0.4"),
               labels_gp = gpar(fontsize=9),
               title_gp = gpar(fontsize=10,fontface="bold"),
               grid_width = unit(2.75, "mm"),
               legend_direction = "vertical",
               legend_width = unit(2, "cm"), 
               fontface="bold", title_position = "topleft"),
             col = colorRamp2(c(-0.2,-0.1,-0.05,-0.025,0,0.0175,0.025,0.05,0.1,0.2,0.4),
                              c("darkblue","#6E7B8B","#6959CD","#836FFF","gray90","#FA8072",
                                "#FF8C69","#CD7054","#EE5C42", "#8B0000","darkred")),
             cell_fun = function(j, i, x, y, width, height, fill)
             {
               grid.text(sprintf("%s", res.plot$res.plot.model.q[i, j]), x, y,
                         gp = gpar(fontsize = 14))
             },
             #left_annotation = ha,
             show_row_dend = TRUE,
             show_column_dend  = FALSE,
             row_names_side = "left",
             clustering_distance_rows = "manhattan",
             clustering_distance_columns = "euclidean",
             rect_gp =gpar(col = "grey90", lwd = 0.5),
             border=T,
             # show_heatmap_legend = FALSE,
             column_title = "\nContinuous\nbiomarkers", 
             column_title_gp = gpar(fill = "lavenderblush1", 
                                    col = "black", border = "black",fontface="bold"),
width = ncol(res.plot$res.plot.model)*unit(7, "mm"), 
height = nrow(res.plot$res.plot.model)*unit(10, "mm")
)

h2 = Heatmap(res.plot$res.plot.model2, 
             column_names_rot =  45,
             column_names_side = "bottom",
             cluster_columns = F,
             cluster_rows = F,
             name="ln(OR)",
             column_names_gp = gpar(fontsize =10),
             row_names_gp = gpar(fontsize = 10),
             heatmap_legend_param = list(
               at=c(0.5,1,1.5,2),
               labels=c("0.5","1","1.5","2"),
               labels_gp = gpar(fontsize=9),
               title_gp = gpar(fontsize=10,fontface="bold"),
               grid_width = unit(2.75, "mm"),
               legend_direction = "vertical",
               legend_width = unit(2, "cm"), 
               title_position = "topleft"),
             col = colorRamp2(c(0.5,1,2),c("darkorchid4","gray90",viridis(2)[2])),
             cell_fun = function(j, i, x, y, width, height, fill,col="grey")
             {
               grid.text(sprintf("%s", res.plot$res.plot.model.q2[i, j]), x, y, gp = gpar(fontsize = 14))
             },
             #left_annotation = ha,
             show_row_dend = FALSE,
             show_column_dend  = FALSE,
             row_names_side = "left",
             clustering_distance_rows = "euclidean",
             clustering_distance_columns = "euclidean",
             rect_gp =gpar(col = "grey90", lwd = 0.5),
             border = T,
             # show_heatmap_legend = FALSE,
             column_title = "\nDiscrete\ntraits", 
             column_title_gp = gpar(fill = "azure1", 
                                    col = "black", border = "black",fontface="bold"),
width = ncol(res.plot$res.plot.model2)*unit(9, "mm"),
height = nrow(res.plot$res.plot.model2)*unit(10, "mm"))

# Combine them using the `+` operator and draw the output
h3=h1+h2

# Generate a PDF containing the image.
pdf(file="//argos.storage.uu.se/MyGroups$/Gold/MolEpi/Personal folders and scripts/ssayols/Uppsala_university/Shafqat/heatmap_liverfatV13.pdf",width = 13,height=8)
draw(h3,heatmap_legend_side="right",legend_grouping = "original") 
dev.off()
