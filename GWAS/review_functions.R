

## Auxiliary functions for review_gwas.R 

# uwemenzel@gmail.com  



##  Find nearest gene to a marker:  

link_nearest_gene <- function(row)  {
  
  marker_chrom = as.integer(row[2]) 		 
  marker_posit = as.integer(row[3]) 		 
  gene_start = gentab[[marker_chrom]]$txStart   
  gene_end = gentab[[marker_chrom]]$txEnd      	   
  gene_positions = gentab[[marker_chrom]]$txMid  
  gene_index = which.min(abs(gene_positions - marker_posit)) 
  gene_row = gentab[[marker_chrom]][gene_index,] 
  res = list()
  gene_name = gene_row[1,2]    				 
  if(gene_name == "") gene_name = gene_row[1,1]
  res["name"] = gene_name  				 
  res["chrom"] = marker_chrom  				
  res["start"] = gene_row[1,4]          		
  res["end"] = gene_row[1,6]            		
  return(res)
}




## Annotation of regions in Manhattan plot

annotateSNPRegions<-function(snps, chr, pos, pvalue, snplist,
	kbaway=0, maxpvalue=1, labels=c(), col=c(), pch=c()) {

	stopifnot(all(length(snps)==length(chr), length(chr)==length(pos),
		length(pos)==length(pvalue)))
	if (length(snplist)==0) stop("snplist vector is empty")

	if(any(pos>1e6)) kbaway<-kbaway*1000

	ann<-rep(0, length(snps))
	for(i in seq_along(snplist)) {
		si<-which(snps==snplist[i])
		ci<-chr[si]
		pi<-pos[si]
		ann[chr==ci & pos >= pi-kbaway & pos <= pi+kbaway & pvalue<=maxpvalue]<-i
	}
	ann<-list(factor(ann, levels=0:length(snplist), labels=c("", snplist)))
	if(length(col)>0 || length(pch)>0 || length(labels)>0) {
		for(i in seq_along(snplist)) {
			ann[[ snplist[i] ]] = list()
			if(length(col)>0) { 
				ann[[ snplist[i] ]]$col = col[ (i-1) %% length(col)+1 ]
			}
			if(length(pch)>0) {
				ann[[ snplist[i] ]]$pch = pch[ (i-1) %% length(pch)+1 ]	
			}
                        if(length(labels)>0) {
                                ann[[ snplist[i] ]]$label = labels[ (i-1) %% length(labels)+1 ]
                        }
		}
	}
	return(ann)
}





##  Manhattan plot, from:  https://genome.sph.umich.edu/wiki/Code_Sample:_Generating_Manhattan_Plots_in_R  

library(lattice)
manhattan.plot<-function(chr, pos, pvalue, 
	sig.level=NA, annotate=NULL, ann.default=list(),
	should.thin=T, thin.pos.places=2, thin.logp.places=2, 
	xlab="Chromosome", ylab=expression(-log[10](p-value)),
	col=c("gray","darkgray"), panel.extra=NULL, pch=20, cex=0.8,...) {

	if (length(chr)==0) stop("chromosome vector is empty")
	if (length(pos)==0) stop("position vector is empty")
	if (length(pvalue)==0) stop("pvalue vector is empty")

	# make sure we have an ordered factor
	if(!is.ordered(chr)) {
		chr <- ordered(chr)
	} else {
		chr <- chr[,drop=T]
	}

	# make sure positions are in kbp
	if (any(pos>1e6)) pos<-pos/1e6;

	# calculate absolute genomic position
	# from relative chromosomal positions
	posmin <- tapply(pos,chr, min);
	posmax <- tapply(pos,chr, max);
	posshift <- head(c(0,cumsum(posmax)),-1);
	names(posshift) <- levels(chr)
	genpos <- pos + posshift[chr];
	getGenPos<-function(cchr, cpos) {
		p<-posshift[as.character(cchr)]+cpos
		return(p)
	}

	# parse annotations
	grp <- NULL
	ann.settings <- list()
	label.default<-list(x="peak",y="peak",adj=NULL, pos=3, offset=0.5, 
		col=NULL, fontface=NULL, fontsize=NULL, show=F)
	parse.label<-function(rawval, groupname) {
		r<-list(text=groupname)
		if(is.logical(rawval)) {
			if(!rawval) {r$show <- F}
		} else if (is.character(rawval) || is.expression(rawval)) {
			if(nchar(rawval)>=1) {
				r$text <- rawval
			}
		} else if (is.list(rawval)) {
			r <- modifyList(r, rawval)
		}
		return(r)
	}

	if(!is.null(annotate)) {
		if (is.list(annotate)) {
			grp <- annotate[[1]]
		} else {
			grp <- annotate
		} 
		if (!is.factor(grp)) {
			grp <- factor(grp)
		}
	} else {
		grp <- factor(rep(1, times=length(pvalue)))
	}
  
	ann.settings<-vector("list", length(levels(grp)))
	ann.settings[[1]]<-list(pch=pch, col=col, cex=cex, fill=col, label=label.default)

	if (length(ann.settings)>1) { 
		lcols<-trellis.par.get("superpose.symbol")$col 
		lfills<-trellis.par.get("superpose.symbol")$fill
		for(i in 2:length(levels(grp))) {
			ann.settings[[i]]<-list(pch=pch, 
				col=lcols[(i-2) %% length(lcols) +1 ], 
				fill=lfills[(i-2) %% length(lfills) +1 ], 
				cex=cex, label=label.default);
			ann.settings[[i]]$label$show <- T
		}
		names(ann.settings)<-levels(grp)
	}
	for(i in 1:length(ann.settings)) {
		if (i>1) {ann.settings[[i]] <- modifyList(ann.settings[[i]], ann.default)}
		ann.settings[[i]]$label <- modifyList(ann.settings[[i]]$label, 
			parse.label(ann.settings[[i]]$label, levels(grp)[i]))
	}
	if(is.list(annotate) && length(annotate)>1) {
		user.cols <- 2:length(annotate)
		ann.cols <- c()
		if(!is.null(names(annotate[-1])) && all(names(annotate[-1])!="")) {
			ann.cols<-match(names(annotate)[-1], names(ann.settings))
		} else {
			ann.cols<-user.cols-1
		}
		for(i in seq_along(user.cols)) {
			if(!is.null(annotate[[user.cols[i]]]$label)) {
				annotate[[user.cols[i]]]$label<-parse.label(annotate[[user.cols[i]]]$label, 
					levels(grp)[ann.cols[i]])
			}
			ann.settings[[ann.cols[i]]]<-modifyList(ann.settings[[ann.cols[i]]], 
				annotate[[user.cols[i]]])
		}
	}
 	rm(annotate)

	# reduce number of points plotted
	if(should.thin) {
		thinned <- unique(data.frame(
			logp=round(-log10(pvalue),thin.logp.places), 
			pos=round(genpos,thin.pos.places), 
			chr=chr,
			grp=grp)
		)
		logp <- thinned$logp
		genpos <- thinned$pos
		chr <- thinned$chr
		grp <- thinned$grp
		rm(thinned)
	} else {
		logp <- -log10(pvalue)
	}
	rm(pos, pvalue)
	gc()

	# custom axis to print chromosome names
	axis.chr <- function(side,...) {
		if(side=="bottom") {
			panel.axis(side=side, outside=T,
				at=((posmax+posmin)/2+posshift),
				labels=levels(chr), 
				ticks=F, rot=0,
				check.overlap=F
			)
		} else if (side=="top" || side=="right") {
			panel.axis(side=side, draw.labels=F, ticks=F);
		}
		else {
			axis.default(side=side,...);
		}
	 }

	# make sure the y-lim covers the range (plus a bit more to look nice)
	prepanel.chr<-function(x,y,...) { 
		A<-list();
		maxy<-ceiling(max(y, ifelse(!is.na(sig.level), -log10(sig.level), 0)))+.5;
		A$ylim=c(0,maxy);
		A;
	}

	xyplot(logp~genpos, chr=chr, groups=grp,
		axis=axis.chr, ann.settings=ann.settings, 
		prepanel=prepanel.chr, scales=list(axs="i"),
		panel=function(x, y, ..., getgenpos) {
			if(!is.na(sig.level)) {
				panel.abline(h=-log10(sig.level), lty=2);
			}
			panel.superpose(x, y, ..., getgenpos=getgenpos);
			if(!is.null(panel.extra)) {
				panel.extra(x,y, getgenpos, ...)
			}
		},
		panel.groups = function(x,y,..., subscripts, group.number) {
			A<-list(...)
			gs <- ann.settings[[group.number]]
			A$col.symbol <- gs$col[(as.numeric(chr[subscripts])-1) %% length(gs$col) + 1]    
			A$cex <- gs$cex[(as.numeric(chr[subscripts])-1) %% length(gs$cex) + 1]
			A$pch <- gs$pch[(as.numeric(chr[subscripts])-1) %% length(gs$pch) + 1]
			A$fill <- gs$fill[(as.numeric(chr[subscripts])-1) %% length(gs$fill) + 1]
			A$x <- x
			A$y <- y
			do.call("panel.xyplot", A)
			# draw labels (if requested)
			if(gs$label$show) {
				gt<-gs$label
				names(gt)[which(names(gt)=="text")]<-"labels"
				gt$show<-NULL
				if(is.character(gt$x) | is.character(gt$y)) {
					peak = which.max(y)
					center = mean(range(x))
					if (is.character(gt$x)) {
						if(gt$x=="peak") {gt$x<-x[peak]}
						if(gt$x=="center") {gt$x<-center}
					}
					if (is.character(gt$y)) {
						if(gt$y=="peak") {gt$y<-y[peak]}
					}
				}
				if(is.list(gt$x)) {
					gt$x<-A$getgenpos(gt$x[[1]],gt$x[[2]])
				}
				do.call("panel.text", gt)
			}
		},
		xlab=xlab, ylab=ylab, 
		panel.extra=panel.extra, getgenpos=getGenPos, ...
	);
}





