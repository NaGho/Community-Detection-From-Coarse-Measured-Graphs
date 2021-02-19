import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import signal


# dist1 = stats.uniform(loc=2, scale=3)
# dist1 = stats.norm(loc=2, scale=2)
# dist1 = stats.chi2(df=2, loc=0, scale=1)
# dist1 = stats.expon(loc=0, scale=1)
# dist1 = stats.gamma(a=2, loc=0, scale=1)
# dist1 = stats.beta(a=2, b=2, loc=0, scale=1)
# dist1 = stats.lognorm(s=0.5, loc=0, scale=1)
dist1 = stats.t(df=3, loc=0, scale=1)

# 
std = 1
dist2 = stats.norm(loc=0, scale=std)



delta = 1e-4
big_grid = np.arange(-10,10,delta)

# 
pmf1 = dist1.pdf(big_grid)*delta
print("Sum of pmf1: "+str(sum(pmf1)))



pmf2 = dist2.pdf(big_grid)*delta
print("Sum of pmf2: "+str(sum(pmf2)))


conv_pmf = signal.fftconvolve(pmf1,pmf2,'same')
print("Sum of convoluted pmf: "+str(sum(conv_pmf)))

pdf1 = pmf1/delta
pdf2 = pmf2/delta
conv_pdf = conv_pmf/delta
print("Integration of convoluted pdf: " + str(np.trapz(conv_pdf, big_grid)))


plt.plot(big_grid,pdf1, label='distribution-1')
plt.plot(big_grid,pdf2, label='distribution-2')
plt.plot(big_grid,conv_pdf, label='Sum')
plt.legend(loc='best'), plt.suptitle('PDFs')
plt.show()