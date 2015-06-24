# 
# functions for performing the projection predictive variable selection
# for linear Gaussian model.
#

lm_proj <- function(w,sigma2,x,indproj) {
	
	# assume the intercept term is stacked into w, and x contains 
	# a corresponding vector of ones. returns the projected samples
	# and estimated kl-divergence.

	# pick the columns of x that form the projection subspace
	n <- dim(x)[1]
	xp <- x[,indproj]

	# solve the projection equations
	fit <- x %*% w # fit of the full model
	wp <- solve(t(xp) %*% xp, t(xp) %*% fit)
	sigma2p <- sigma2 + colMeans((fit - xp %*% wp)^2)
	
	# this is the estimated kl-divergence between the full and projected model
	kl <- mean(0.5*log(sigma2p/sigma2))
	
	# reshape wp so that it has same dimensionality as x, and zeros for
	# those variables that are not included in the projected model
	d <- dim(w)[1]
	S <- dim(w)[2]
	wptemp <- matrix(0, d, S)
	wptemp[indproj,] <- wp
	wp <- wptemp
	
	return(list(w=wp, sigma2=sigma2p, kl=kl))
}


lm_fprojsel <- function(w, sigma2, x) {

	# forward variable selection using the projection
	
	d = dim(x)[2]
	chosen <- 1 # chosen variables, start from the model with the intercept only
	notchosen <- setdiff(1:d, chosen)
	
	# start from the model having only the intercept term
	kl <- rep(0,d)
	kl[1] <- lm_proj(w,sigma2,x,1)$kl

	# start adding variables one at a time
	for (k in 2:d) {
	
		nleft <- length(notchosen)
		val <- rep(0, nleft)

		for (i in 1:nleft) {
			ind <- sort( c(chosen, notchosen[i]) )
			proj <- lm_proj(w,sigma2,x,ind)
			val[i] <- proj$kl
		}

		# find the variable that minimizes the kl
		imin <- which.min(val)
		chosen <- c(chosen, notchosen[imin])
		notchosen <- setdiff(1:d, chosen)
	
		kl[k] <- val[imin]
	}
	return(list(chosen=chosen, kl=kl))
}
