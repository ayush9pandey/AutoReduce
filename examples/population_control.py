n = 8 # Number of states 

from SBMLReduce import *
sr = SBMLReduce()

# Either
# sr.load_SBML_model('my_sbml_model.xml')

# OR, write ODEs
#      x = 0, T1, 1, A1, 2, S1, 3, S2, 4, T2, 5, A2, 6, C1, 7, C2
#      P = 0, beta_S1, 1, l_S1, 2, K_S1, 3, kb, 4, beta_S2, 5, l_S2, 6,
#      K_S2, 7, beta_lac, 8, l_lac, 9, K_lac, 10, beta_tet, 11, l_tet, 12,
#      K_tet, 13, kc, 14, C_max, 15, dc, 16, xx, 17, I, 18, xx, 19, atc, 20,K_tox

# parameter values
P = np.zeros(24)
P[0] = 6
P[1] = 2e-3
P[2] = 430
P[3] = 30
P[4] = 6
P[5] = 2e-3
P[6] = 190
P[7] = 19.8e-3
P[8] = 1.5e-3
P[9] = 1.4e5
P[10] = 14.4e-3
P[11] = 2.1e-4
P[12] = 13
P[13] = 0.6
P[14] = 5500
P[15] = 0.8
P[16] = np.Inf
P[17] = 1e6
P[18] = np.Inf 
P[19] = 324
P[20] = 1
P[21] = 0.1
P[22] = 1.5
P[23] = 0.5
sr.params_values = P.copy()

sr.timepoints = np.linspace(0, 40, 100) # timepoints for simulation
x_init = np.zeros(n) # Initial conditions
x_init[6] = 100
x_init[7] = 500
sr.x_init = x_init

sr.error_tol = 1000
sr.nstates_tol = 5
x, f, P = sr.load_ODE_model(n, len(sr.params_values))
params = P
# T1 and A1
f[0] = P[0]*(P[1] + x[2]**2/(P[2]+x[2]**2)) - P[3]*x[0]*x[1] - P[22] * x[0]
f[1] = 5*P[4]*(P[5] + x[3]**2/(P[6]+x[3]**2)) - P[22] * x[1] - P[3]*x[0]*x[1]

# f[0] = P[0]*(x[2]**2/(P[2]+x[2]**2)) - P[3]*x[0]*x[1]
# f[1] = P[4]*(x[3]**2/(P[6]+x[3]**2)) - P[3]*x[0]*x[1]

#  S1 and S2 (scaled with cell count)
f[2] = P[7]*(P[8] + P[17]**2/(P[9]+P[17]**2))*x[6] - P[23] * x[2]
f[3] = P[10]*(P[11] + P[19]**2/(P[12]+P[19]**2))*x[7] - P[23] * x[3]

# f[2] = P[7]*(P[17]**2/(P[9]+P[17]**2))*x[6] - P[23] * x[2]
# f[3] = P[10]*(P[19]**2/(P[12]+P[19]**2))*x[7] - P[23] * x[3]

#  T2 and A2
f[4] = P[4]*(P[5] + x[3]**2/(P[6]+x[3]**2)) - P[3]*x[4]*x[5] - P[22] * x[4]
f[5] = 5*P[0]*(P[1] + x[2]**2/(P[2]+x[2]**2)) - P[22] * x[5]-P[3]*x[4]*x[5]

# f[4] = P[4]*(x[3]**2/(P[6]+x[3]**2)) - P[3]*x[4]*x[5] - P[22] * x[4]
# f[5] = P[0]*(x[2]**2/(P[2]+x[2]**2)) - P[22] * x[5]-P[3]*x[4]*x[5]

#  Cell 1 and Cell 2
f[6] = P[13]*(1 - (x[6] + x[7])/P[14])*x[6] - P[15]*x[6]*(x[0]/(P[20] + x[0])) - P[21] * x[6]
f[7] = P[13]*(1 - (x[6] + x[7])/P[14])*x[7] - P[15]*x[7]*(x[4]/(P[20] + x[4])) - P[21] * x[7]

C = np.zeros((2,len(x)), dtype=int)
C[0][6] = 1
C[1][7] = 1
C = C.tolist()
sr.C = C
sr.compute_reduced_model()
f_hat = sr.get_reduced_model('Sympy')
print(f_hat)
# f_hat_SBMLDocument = sr.get_reduced_model('SBML')
# f_hat_Scipy_callable_object = sr.get_reduced_model('ODE')
