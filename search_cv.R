#
# cross-validation of the variable searching. fits the full model K=10 times
# and does the variable selection each time separately, while measuring the performance
# of the found models on the validation data.
#

# load the data
n <- 1000 # number of training points
data <- as.matrix(read.table('crimedata.csv', sep=','))
d <- dim(data)[2]-1

# training data
y <- data[1:n, 1]
x <- data[1:n, 2:(d+1)]


# cross-validate the variable searching (10-fold)
cvk <- 10
lpd <- matrix(0, cvk, d+1) # lpd for each validation set
mse <- matrix(0, cvk, d+1) # mse for - '' -

fit <- list()
spath <- list()

for (i in 1:cvk) {

	# form the training and validation sets
	ival <- seq(i,n,cvk)
	itr <- setdiff(1:n,ival)
	nval <- length(ival)
	ntr <- length(itr)
	
	# fit the full model (recompilation could be avoided by using the
	# same fit-object each time)
	datacv <- list(X=x[itr,], y=y[itr], n=ntr, d=d, nu=3.0)
	fit[[i]] <- stan('lg_t.stan', data=datacv, iter=1000, chains=4)
	e <- extract(fit[[i]])
	
	# perform the variable selection
	w <- rbind(e$w0, t(e$w)) # stack the intercept and the input weights
	sigma2 <- e$sigma^2
	xtr <- cbind(rep(1,ntr), x[itr,]) # add a vector of ones to the input matrix
	spath[[i]] <- lm_fprojsel(w, sigma2, xtr)	
	
	# make predictions for the observations in the validation set
	xval <- cbind(rep(1, nval), x[ival,])
	yval <- y[ival]
	for (k in 1:(d+1)) {

		# projected parameters
		submodel <- lm_proj(w, sigma2, xtr, spath[[i]]$chosen[1:k])
		wp <- submodel$w
		sigma2p <- submodel$sigma2
		
		# mean squared error
		ypred <- rowMeans(xval %*% wp)
		mse[i,k] <- mean((yval-ypred)^2)
		
		# mean log predictive density using the projected parameters
		pd <- dnorm(yval, xval %*% wp, sqrt(sigma2p))
		lpd[i,k] <- mean(log(rowMeans(pd)))
	}
}

