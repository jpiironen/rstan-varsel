#
# perform the variable selection using the training data, and measure the
# performance of the found models on the test data.
#

# load the data
n <- 1000 # number of training points
data <- as.matrix(read.table('crimedata.csv', sep=','))
dims <- dim(data)
ntotal <- dims[1]
nt <- ntotal - n # number of test points
d <- dims[2]-1

# training data
y <- data[1:n, 1]
x <- data[1:n, 2:(d+1)]

# test data
yt <- data[(n+1):ntotal, 1]
xt <- data[(n+1):ntotal, 2:(d+1)]

# fit the full model
fit <- stan("lg_t.stan", data=list(X=x,y=y,d=d,n=n,nu=3.0), iter=1000, chains=4)
e <- extract(fit)

w <- rbind(e$w0, t(e$w)) # stack the intercept and weights
sigma2 <- e$sigma^2
x <- cbind(rep(1,n), x) # add a vector of ones to the predictor matrix
xt <- cbind(rep(1,nt), xt) 

# perform the variable selection
spath <- lm_fprojsel(w, sigma2, x)

# compute the predictions on the test set using the projected parameters
mlpd <- rep(0, d+1)
mse <- rep(0, d+1)
for (k in 1:(d+1)) {

	# projected parameters
	submodel <- lm_proj(w, sigma2, x, spath$chosen[1:k])
	wp <- submodel$w
	sigma2p <- rep(submodel$sigma2, each=nt)
		
	# mean squared error
	ypred <- rowMeans(xt %*% wp)
	mse[k] <- mean((yt-ypred)^2)
	
	# mean log predictive density
	pd <- dnorm(yt, xt %*% wp, sqrt(sigma2p))
	mlpd[k] <- mean(log(rowMeans(pd)))
}
