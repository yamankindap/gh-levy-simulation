import numpy as np
from scipy.special import gamma as gammafnc
from scipy.special import gammainc, gammaincc, gammaincinv
from scipy.special import kv
from scipy.special import hankel1, hankel2

def incgammau(s, x):
    return gammaincc(s,x)*gammafnc(s)

def incgammal(s, x):
    return gammainc(s,x)*gammafnc(s)

class LevyProcess:

    def integral(self, evaluation_points, t_series, x_series):
        W = [x_series[t_series<point].sum() for point in evaluation_points]
        return np.array(W).T

class GammaProcess(LevyProcess):

    def __init__(self, beta, C):
        self.beta = beta
        self.C = C

    def set_parameters(self, beta, C):
        self.beta = beta
        self.C = C

    def h_gamma(self, gamma):
        return 1/(self.beta*(np.exp(gamma/self.C)-1))

    def simulate_jumps(self, rate=1.0, M=100, gamma_0=0.0):
        gamma_sequence = np.random.exponential(scale=1/rate, size=M)
        gamma_sequence[0] += gamma_0 
        gamma_sequence = gamma_sequence.cumsum()
        x_series = self.h_gamma(gamma_sequence)
        thinning_function = (1+self.beta*x_series)*np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < thinning_function]
        return gamma_sequence, x_series

    def unit_expected_residual(self, c):
        return (self.C/self.beta)*incgammal(1, self.beta*c)

    def unit_variance_residual(self, c):
        return (self.C/self.beta**2)*incgammal(2, self.beta*c)

class StableProcess(LevyProcess):
    def __init__(self, alpha, C):
        self.alpha = alpha
        self.C = C

    def set_parameters(self, alpha, C):
        self.alpha = alpha
        self.C = C

    def check_parameter_constraints(self):
        if (self.alpha >= 1):
            raise ValueError('The alpha parameter is set to greater than or equal to 1.')

    def h_stable(self, gamma):
        return np.power((self.alpha/self.C)*gamma, np.divide(-1,self.alpha))

    def simulate_jumps(self, rate=1.0, M=1000, gamma_0=0.0):
        gamma_sequence = np.random.exponential(scale=1/rate, size=M)
        gamma_sequence[0] += gamma_0 
        gamma_sequence = gamma_sequence.cumsum()
        x_series = self.h_stable(gamma_sequence)
        return gamma_sequence, x_series

    def unit_expected_residual(self, c):
        return (self.C/(1-self.alpha))*(c**(1-self.alpha))

    def unit_variance_residual(self, c):
        return (self.C/(2-self.alpha))*(c**(2-self.alpha))

class TemperedStableProcess(LevyProcess):

    def __init__(self, alpha, beta, C):
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def set_parameters(self, alpha, beta, C):
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def h_stable(self, gamma):
        return np.power((self.alpha/self.C)*gamma, np.divide(-1,self.alpha))

    def simulate_jumps(self, rate=1.0, M=1000, gamma_0=0.0):
        gamma_sequence = np.random.exponential(scale=1/rate, size=M)
        gamma_sequence[0] += gamma_0 
        gamma_sequence = gamma_sequence.cumsum()
        x_series = self.h_stable(gamma_sequence)
        thinning_function = np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < thinning_function]
        return gamma_sequence, x_series

    def unit_expected_residual(self, c):
        return (self.C*self.beta**(self.alpha-1))*incgammal(1-self.alpha, self.beta*c)

    def unit_variance_residual(self, c):
        return (self.C*self.beta**(self.alpha-2))*incgammal(2-self.alpha, self.beta*c)


class GeneralisedInverseGaussianProcess(LevyProcess):

    def __init__(self, lam, gamma, delta, rate=1.0, M_gamma=10, M_stable=100, tolerance=None, pt=0.05):
        # Process parameters:
        self.lam = lam
        self.gamma = gamma
        self.delta = delta
        self.abs_lam = np.abs(lam)

        # Simulation parameters and sub-processes:
        self.set_simulation_parameters(rate=rate, M_gamma=M_gamma, M_stable=M_stable, tolerance=tolerance, pt=pt)

        self.gamma_process = GammaProcess(None, None)
        self.gamma_process2 = GammaProcess(None, None)
        self.tempered_stable_process = TemperedStableProcess(None, None, None)

        # Define a third gamma process for the positive lam extension
        if (self.lam > 0):
            self.pos_ext_gamma_process = GammaProcess(None, None)
            C = self.lam
            beta = 0.5*self.gamma**2
            self.pos_ext_gamma_process.set_parameters(beta, C)

        self.set_simulation_method()

    def set_simulation_parameters(self, rate, M_gamma, M_stable, tolerance, pt):
        self.rate = rate
        self.M_gamma = M_gamma
        self.M_stable = M_stable
        self.pt = pt

        self.max_iter = int(10000/5)

        if tolerance is None:
            if self.abs_lam > 1:
                self.tolerance = 0.1
            else:
                self.tolerance = 0.01
        else:
            self.tolerance = tolerance

    # Auxiliary functionality
    def cornerpoint(self):
        return np.power(np.power(float(2), 1-2*self.abs_lam)*np.pi/np.power(gammafnc(self.abs_lam), 2), 1/(1-2*self.abs_lam))

    def H_squared(self, z):
        return np.real(hankel1(self.abs_lam, z)*hankel2(self.abs_lam, z))

    def probability_density(self, x):
        return np.power(self.gamma/self.delta, self.lam)*(1/(2*kv(self.lam, self.delta*self.gamma))*np.power(x, self.lam-1)*np.exp(-(self.gamma**2*x+self.delta**2/x)/2))

    def random_sample(self, size):
        def thinning_function(delta, x):
            return np.exp(-(1/2)*(np.power(delta, 2)*(1/x)))
        def reciprocal_sample(x, i):
            return x**(i)
        def random_GIG(lam, gamma, delta, size=1):
            i = 1
            if lam < 0:
                tmp = gamma
                gamma = delta
                delta = tmp
                lam = -lam
                i = -1
            shape = lam
            scale = 2/np.power(gamma, 2)
            gamma_rv = np.random.gamma(shape=shape, scale=scale, size=size)
            u = np.random.uniform(low=0.0, high=1.0, size=size)
            sample = gamma_rv[u < thinning_function(delta, gamma_rv)]
            return reciprocal_sample(sample, i)
        sample = np.array([])
        while sample.size < size:
            sample = np.concatenate((sample, random_GIG(self.lam, self.gamma, self.delta, size=size)))
        return sample[np.random.randint(low=0, high=sample.size, size=size)]

    # Select simulation method and set corresponding parameters
    def _simulate_with_positive_extension(self):
        x_series = self.simulate_Q_GIG()
        x_P_series = self.simulate_adaptive_positive_extension_series()
        return np.concatenate((x_series, x_P_series))

    def set_simulation_method(self, method=None):
        # Automatically select a method for simulation
        if method is None:
            if (self.abs_lam >= 0.5):
                if (self.gamma == 0) or (self.abs_lam == 0.5):
                    print('Simulation method is set to GIG paper version.')
                    # Set parameters of the tempered stable process...
                    alpha = 0.5
                    C = self.delta*gammafnc(0.5)/(np.sqrt(2)*np.pi)
                    beta = 0.5*self.gamma**2

                    if (self.gamma == 0):
                        print('The dominating point process is set as a stable process.')
                        self.tempered_stable_process = StableProcess(alpha=alpha, C=C)
                    else:
                        self.tempered_stable_process.set_parameters(alpha, beta, C)

                    self.simulate_Q_GIG = self.simulate_adaptive_series_setting_1
                    if (self.lam > 0):
                        print('An independent gamma process extension will be made.')
                        self.simulate_jumps = self._simulate_with_positive_extension
                    else:
                        self.simulate_jumps = self.simulate_Q_GIG
                else:
                    print('Simulation method is set to improved version.')
                    # Set parameters of the two gamma and one TS processes...
                    z1 = self.cornerpoint()
                    C1 = z1/(np.pi*self.abs_lam*2*(1+self.abs_lam))
                    beta1 = 0.5*self.gamma**2
                    self.gamma_process.set_parameters(beta1, C1)
                    C2 = z1/(np.pi*2*(1+self.abs_lam))
                    beta2 = 0.5*self.gamma**2 + (z1**2)/(2*self.delta**2)
                    self.gamma_process2.set_parameters(beta2, C2)
                    C = self.delta/(np.sqrt(2*np.pi))
                    alpha = 0.5
                    beta = 0.5*self.gamma**2 + (z1**2)/(2*self.delta**2)
                    self.tempered_stable_process.set_parameters(alpha, beta, C)

                    self.simulate_Q_GIG = self.simulate_adaptive_combined_series_setting_1
                    if (self.lam > 0):
                        self.simulate_jumps = self._simulate_with_positive_extension
                    else:
                        self.simulate_jumps = self.simulate_Q_GIG
            else:
                print('Simulation method is set to improved version for 0 < |lam| < 0.5.')
                # Set parameters of the two gamma and one TS processes...
                z0 = self.cornerpoint()
                H0 = z0*self.H_squared(z0)
                C1 = z0/((np.pi**2)*H0*self.abs_lam*(1+self.abs_lam))
                beta1 = 0.5*self.gamma**2
                self.gamma_process.set_parameters(beta1, C1)
                C2 = z0/((np.pi**2)*(1+self.abs_lam)*H0)
                beta2 = 0.5*self.gamma**2 + (z0**2)/(2*self.delta**2)
                self.gamma_process2.set_parameters(beta2, C2)
                C = np.sqrt(2*self.delta**2)*gammafnc(0.5)/(H0*np.pi**2)
                alpha = 0.5
                beta = 0.5*self.gamma**2
                self.tempered_stable_process.set_parameters(alpha, beta, C)

                self.simulate_Q_GIG = self.simulate_adaptive_combined_series_setting_2
                if (self.lam > 0):
                    self.simulate_jumps = self._simulate_with_positive_extension
                else:
                    self.simulate_jumps = self.simulate_Q_GIG
        else:
            raise ValueError('The manual selection functionality for simulation method is NOT implemented.')
        
    # Positive lam extension module
    def simulate_adaptive_positive_extension_series(self):
        gamma_sequence, x_series = self.pos_ext_gamma_process.simulate_jumps(rate=self.rate, M=self.M_gamma, gamma_0=0)

        truncation_level = self.pos_ext_gamma_process.h_gamma(gamma_sequence[-1])
        residual_expected_value = self.rate*self.pos_ext_gamma_process.unit_expected_residual(truncation_level)
        residual_variance = self.rate*self.pos_ext_gamma_process.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum()
        while (residual_variance/((E_c - residual_expected_value)**2) > self.pt) or (E_c < residual_expected_value):
            gamma_sequence_extension, x_series_extension = self.pos_ext_gamma_process.simulate_jumps(rate=self.rate, M=self.M_gamma, gamma_0=gamma_sequence[-1])
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension))
            x_series = np.concatenate((x_series, x_series_extension))

            truncation_level = self.pos_ext_gamma_process.h_gamma(gamma_sequence[-1])
            residual_expected_value = self.rate*self.pos_ext_gamma_process.unit_expected_residual(truncation_level)
            residual_variance = self.rate*self.pos_ext_gamma_process.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum()
        # We do not use residual approximation in this setting since Asmussen and Rosinski 2001 shows it is not valid for the gamma process.
        # It is also likely not necessary since the jumps from a gamma process rapidly converge to zero.
        return x_series

    # Jump magnitude simulation:
    ## GIG-paper:
    def simulate_adaptive_series_setting_1(self):
        gamma_sequence, x_series = self.simulate_series_setting_1(rate=self.rate, M=self.M_stable, gamma_0=0.0)
        truncation_level = self.tempered_stable_process.h_stable(gamma_sequence[-1])
        residual_expected_value = self.rate*self.tempered_stable_process.unit_expected_residual(truncation_level)
        residual_variance = self.rate*self.tempered_stable_process.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum()
        while (residual_variance/((E_c - residual_expected_value)**2) > self.pt) or (E_c < residual_expected_value):
            gamma_sequence_extension, x_series_extension = self.simulate_series_setting_1(rate=self.rate, M=self.M_stable, gamma_0=gamma_sequence[-1])
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension))
            x_series = np.concatenate((x_series, x_series_extension))
            truncation_level = self.tempered_stable_process.h_stable(gamma_sequence[-1])
            residual_expected_value = self.rate*self.tempered_stable_process.unit_expected_residual(x_series[-1])
            residual_variance = self.rate*self.tempered_stable_process.unit_variance_residual(x_series[-1])
            E_c = self.tolerance*x_series.sum()

        return x_series, truncation_level

    def simulate_series_setting_1(self, rate, M, gamma_0=0.0):
        gamma_sequence, x_series = self.tempered_stable_process.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)
        z_series = np.sqrt(np.random.gamma(shape=0.5, scale=np.power(x_series/(2*self.delta**2), -1.0)))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*z_series*np.pi)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < acceptance_prob]
        return gamma_sequence, x_series

    ## GH-paper:
    def simulate_adaptive_combined_series_setting_1(self):
        # The smallest simulated jump from each point process can be accessed through the corresponding h(gamma[-1])
        # x_series in this case only contains 'accepted' jumps...

        # Simulate jump magnitudes:
        gamma_sequence_N_Ga_1, x_series_N_Ga_1 = self.simulate_left_bounding_series_setting_1(rate=self.rate, M=self.M_gamma)
        gamma_sequence_N_Ga_2, x_series_N_Ga_2 = self.simulate_left_bounding_series_setting_1_alternative(rate=self.rate, M=self.M_gamma)
        gamma_sequence_N2, x_series_N2 = self.simulate_right_bounding_series_setting_1(rate=self.rate, M=self.M_stable)
        x_series = np.concatenate((x_series_N_Ga_1, x_series_N_Ga_2))
        x_series = np.concatenate((x_series, x_series_N2))

        truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[-1])
        truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[-1])
        truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[-1])

        # Residual statistics:
        residual_expected_value_N_Ga_1 = self.rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
        residual_variance_N_Ga_1 = self.rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
        residual_expected_value_N_Ga_2 = self.rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
        residual_variance_N_Ga_2 = self.rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
        residual_expected_value_N2 = self.rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
        residual_variance_N2 = self.rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
        residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
        residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

        ## Select process for simulation:
        selection = np.argmax([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2])

        # Adaptive simulation:
        E_c = self.tolerance*x_series.sum()
        itr = 1
        while (residual_variance/((E_c - residual_expected_value)**2) > self.pt) or (E_c < residual_expected_value):
            
            # Debug code:
            if (itr > self.max_iter):
                print('Max iteration reached.')
                break

            if (selection == 2):
                gamma_sequence_extension, x_series_extension = self.simulate_right_bounding_series_setting_1(rate=self.rate, M=self.M_stable, gamma_0=gamma_sequence_N2[-1])
                gamma_sequence_N2 = np.concatenate((gamma_sequence_N2, gamma_sequence_extension))
                x_series_N2 = np.concatenate((x_series_N2, x_series_extension))
                truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[-1])
                residual_expected_value_N2 = self.rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
                residual_variance_N2 = self.rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
            elif (selection == 0):
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_1(rate=self.rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_1[-1])
                gamma_sequence_N_Ga_1 = np.concatenate((gamma_sequence_N_Ga_1, gamma_sequence_extension))
                x_series_N_Ga_1 = np.concatenate((x_series_N_Ga_1, x_series_extension))
                truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[-1])
                residual_expected_value_N_Ga_1 = self.rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
                residual_variance_N_Ga_1 = self.rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
            else:
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_1_alternative(rate=self.rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_2[-1])
                gamma_sequence_N_Ga_2 = np.concatenate((gamma_sequence_N_Ga_2, gamma_sequence_extension))
                x_series_N_Ga_2 = np.concatenate((x_series_N_Ga_2, x_series_extension))
                truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[-1])
                residual_expected_value_N_Ga_2 = self.rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
                residual_variance_N_Ga_2 = self.rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
            
            x_series = np.concatenate((x_series, x_series_extension))
            
            residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
            residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

            selection = np.argmax([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2])

            E_c = self.tolerance*x_series.sum()
            itr += 1

        truncation_level = truncation_level_N2
        return x_series, truncation_level

    def simulate_left_bounding_series_setting_1(self, rate, M, gamma_0=0.0):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.gamma_process.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)*incgammal(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)
                        /((x_series**self.abs_lam)*(z1**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z1**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < envelope_fnc]
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)    
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*gammainc(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*self.abs_lam))/(z1**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_left_bounding_series_setting_1_alternative(self, rate, M, gamma_0=0.0):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.gamma_process2.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)*incgammal(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)
                        /((x_series**self.abs_lam)*(z1**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z1**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < envelope_fnc]
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)    
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*gammainc(self.abs_lam, (z1**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*self.abs_lam))/(z1**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_right_bounding_series_setting_1(self, rate, M, gamma_0=0.0):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.tempered_stable_process.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)
        envelope_fnc = incgammau(0.5, (z1**2)*x_series/(2*self.delta**2))/(np.sqrt(np.pi)*np.exp(-(z1**2)*x_series/(2*self.delta**2)))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < envelope_fnc]
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(0.5, u_z*(gammaincc(0.5, (z1**2)*x_series/(2*self.delta**2)))
                                                            + gammainc(0.5, (z1**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = 2/(hankel_squared*z_series*np.pi)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < acceptance_prob]
        return gamma_sequence, x_series


    def simulate_adaptive_combined_series_setting_2(self):
        # Simulate jump magnitudes:
        gamma_sequence_N_Ga_1, x_series_N_Ga_1 = self.simulate_left_bounding_series_setting_2(rate=self.rate, M=self.M_gamma)
        gamma_sequence_N_Ga_2, x_series_N_Ga_2 = self.simulate_left_bounding_series_setting_2_alternative(rate=self.rate, M=self.M_gamma)
        gamma_sequence_N2, x_series_N2 = self.simulate_right_bounding_series_setting_2(rate=self.rate, M=self.M_stable)
        x_series = np.concatenate((x_series_N_Ga_1, x_series_N_Ga_2))
        x_series = np.concatenate((x_series, x_series_N2))

        truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[-1])
        truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[-1])
        truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[-1])

        # Residual statistics:
        residual_expected_value_N_Ga_1 = self.rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
        residual_variance_N_Ga_1 = self.rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
        residual_expected_value_N_Ga_2 = self.rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
        residual_variance_N_Ga_2 = self.rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
        residual_expected_value_N2 = self.rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
        residual_variance_N2 = self.rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
        residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
        residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

        ## Select process for simulation:
        selection = np.argmax([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2])

        # Adaptive simulation:
        E_c = self.tolerance*x_series.sum()
        itr = 1
        while (residual_variance/((E_c - residual_expected_value)**2) > self.pt) or (E_c < residual_expected_value):
            
            # Debug code:
            if (itr > self.max_iter):
                print('Max iteration reached.')
                break

            if (selection == 2):
                gamma_sequence_extension, x_series_extension = self.simulate_right_bounding_series_setting_2(rate=self.rate, M=self.M_stable, gamma_0=gamma_sequence_N2[-1])
                gamma_sequence_N2 = np.concatenate((gamma_sequence_N2, gamma_sequence_extension))
                x_series_N2 = np.concatenate((x_series_N2, x_series_extension))
                truncation_level_N2 = self.tempered_stable_process.h_stable(gamma_sequence_N2[-1])
                residual_expected_value_N2 = self.rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
                residual_variance_N2 = self.rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
            elif (selection == 0):
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_2(rate=self.rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_1[-1])
                gamma_sequence_N_Ga_1 = np.concatenate((gamma_sequence_N_Ga_1, gamma_sequence_extension))
                x_series_N_Ga_1 = np.concatenate((x_series_N_Ga_1, x_series_extension))
                truncation_level_N_Ga_1 = self.gamma_process.h_gamma(gamma_sequence_N_Ga_1[-1])
                residual_expected_value_N_Ga_1 = self.rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
                residual_variance_N_Ga_1 = self.rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
            else:
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_2_alternative(rate=self.rate, M=self.M_gamma, gamma_0=gamma_sequence_N_Ga_2[-1])
                gamma_sequence_N_Ga_2 = np.concatenate((gamma_sequence_N_Ga_2, gamma_sequence_extension))
                x_series_N_Ga_2 = np.concatenate((x_series_N_Ga_2, x_series_extension))
                truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(gamma_sequence_N_Ga_2[-1])
                residual_expected_value_N_Ga_2 = self.rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
                residual_variance_N_Ga_2 = self.rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
            
            x_series = np.concatenate((x_series, x_series_extension))
            
            residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
            residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

            selection = np.argmax([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2])

            E_c = self.tolerance*x_series.sum()
            itr += 1


        truncation_level = truncation_level_N2
        return x_series, truncation_level

    def simulate_left_bounding_series_setting_2(self, rate, M, gamma_0=0.0):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.gamma_process.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)* incgammal(self.abs_lam, (z0**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)/
            ((x_series**self.abs_lam)*(z0**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z0**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < envelope_fnc]

        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*(gammainc(self.abs_lam, (z0**2)*x_series/(2*self.delta**2)))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = H0/(hankel_squared*((z_series**(2*self.abs_lam))/(z0**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < acceptance_prob]
        return gamma_sequence, x_series
    
    def simulate_left_bounding_series_setting_2_alternative(self, rate, M, gamma_0=0.0):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.gamma_process2.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)
        envelope_fnc = (((2*self.delta**2)**self.abs_lam)* incgammal(self.abs_lam, (z0**2)*x_series/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)/
            ((x_series**self.abs_lam)*(z0**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z0**2)*x_series/(2*self.delta**2)))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < envelope_fnc]

        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(self.abs_lam, u_z*(gammainc(self.abs_lam, (z0**2)*x_series/(2*self.delta**2)))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = H0/(hankel_squared*((z_series**(2*self.abs_lam))/(z0**(2*self.abs_lam-1))))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_right_bounding_series_setting_2(self, rate, M, gamma_0=0.0):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.tempered_stable_process.simulate_jumps(rate=rate, M=M, gamma_0=gamma_0)
        envelope_fnc = gammaincc(0.5, (z0**2)*x_series/(2*self.delta**2))
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < envelope_fnc]
        u_z = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        z_series = np.sqrt(((2*self.delta**2)/x_series)*gammaincinv(0.5, u_z*(gammaincc(0.5, (z0**2)*x_series/(2*self.delta**2)))
                                                            +gammainc(0.5, (z0**2)*x_series/(2*self.delta**2))))
        hankel_squared = self.H_squared(z_series)
        acceptance_prob = H0/(hankel_squared*z_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.size)
        x_series = x_series[u < acceptance_prob]
        return gamma_sequence, x_series

class GeneralisedHyperbolic(LevyProcess):

    def __init__(self, lam, gamma, delta, beta, sigma, rate=1.0, M_gamma=10, M_stable=100, tolerance=None, pt=0.05):
        # Input metrics:
        self.lam = lam
        self.gamma = gamma
        self.delta = delta
        self.beta = beta
        self.sigma = sigma

        # Calculated metrics:
        self.abs_lam = np.abs(lam)

        # Set simulation parameters:
        self.set_simulation_parameters(rate=rate, M_gamma=M_gamma, M_stable=M_stable, tolerance=tolerance, pt=pt)

        # Create subordinator process:
        self.gig_process = GeneralisedInverseGaussianProcess(lam=self.lam,
                                                             gamma=self.gamma,
                                                             delta=self.delta,
                                                             rate=self.rate,
                                                             M_gamma=self.M_gamma,
                                                             M_stable=self.M_stable,
                                                             tolerance=self.tolerance,
                                                             pt=self.pt)

        # Residual approximation:
        self.set_residual_approximation_method()

    def set_simulation_parameters(self, rate, M_gamma, M_stable, tolerance, pt):
        self.rate = rate
        self.M_gamma = M_gamma
        self.M_stable = M_stable
        self.pt = pt

        if tolerance is None:
            if self.abs_lam > 1:
                self.tolerance = 0.1
            else:
                self.tolerance = 0.01
            print('The tolerance parameter of the adaptive truncation process is set to {}'.format(self.tolerance))
        else:
            self.tolerance = tolerance
            print('The tolerance parameter of the adaptive truncation process is set to {}'.format(self.tolerance))

    def random_sample(self, size):
        gig_sample = self.gig_process.random_sample(size=size)
        gh_sample = self.beta*gig_sample + np.sqrt(gig_sample)*np.random.randn(gig_sample.size)
        return gh_sample

    # Residual approximation module
    def _simulate_exact(self, truncation_level_gamma, truncation_level_TS, size): # Here we're not using truncation_level_gamma argument...
        residual_expected_value_GIG = self.rate*self.gig_process.tempered_stable_process.unit_expected_residual(truncation_level_TS)
        residual_variance_GIG = self.rate*self.gig_process.tempered_stable_process.unit_variance_residual(truncation_level_TS)
        #residual_gaussian = np.random.normal(loc=residual_expected_value_GIG, scale=np.sqrt(residual_variance_GIG))
        residual_expected_value_GH = (self.beta*residual_expected_value_GIG)/size
        residual_variance_GH = ((self.beta**2)*residual_variance_GIG + (self.sigma**2)*residual_expected_value_GIG)/size
        residual_gaussians = np.random.normal(loc=residual_expected_value_GH, scale=np.sqrt(residual_variance_GH), size=size)
        return residual_gaussians

    def _simulate_lower_bound(self, truncation_level_gamma, truncation_level_TS, size):
        residual_expected_value_GIG = self.rate*self.lb_gamma_process.unit_expected_residual(truncation_level_gamma) + self.rate*self.lb_tempered_stable_process.unit_expected_residual(truncation_level_TS)
        residual_variance_GIG = self.rate*self.lb_gamma_process.unit_variance_residual(truncation_level_gamma) + self.rate*self.lb_tempered_stable_process.unit_variance_residual(truncation_level_TS)
        #residual_gaussian = np.random.normal(loc=residual_expected_value_GIG, scale=np.sqrt(residual_variance_GIG))
        residual_expected_value_GH = (self.beta*residual_expected_value_GIG)/size
        residual_variance_GH = ((self.beta**2)*residual_variance_GIG + (self.sigma**2)*residual_expected_value_GIG)/size
        residual_gaussians = np.random.normal(loc=residual_expected_value_GH, scale=np.sqrt(residual_variance_GH), size=size)
        return residual_gaussians

    def set_residual_approximation_method(self):
        if (self.abs_lam >= 0.5):
            if (self.abs_lam == 0.5):
                print('Residual approximation method is set to exact method.')
                self.residual_gaussian_sequence = self._simulate_exact
            else:
                # Initialise the lower bounding point processes for residual approximation
                print('Residual approximation method is set to lower bounding method.')
                z0 = self.gig_process.cornerpoint()
                H0 = z0*self.gig_process.H_squared(z0)
                C_gamma_B = z0/((np.pi**2)*H0*self.abs_lam)
                beta_gamma_B = 0.5*self.gamma**2 + (self.abs_lam/(1+self.abs_lam))*(z0**2)/(2*self.delta**2)
                self.lb_gamma_process = GammaProcess(beta_gamma_B, C_gamma_B)
                beta_0 = 1.95 # This parameter value can be optimised further in the future...
                C_TS_B = (2*self.delta*np.sqrt(np.e)*np.sqrt(beta_0-1))/((np.pi**2)*H0*beta_0)
                beta_TS_B = 0.5*self.gamma**2 + (beta_0*z0**2)/(2*self.delta**2)
                self.lb_tempered_stable_process = TemperedStableProcess(0.5, beta_TS_B, C_TS_B)
                # Select the appropriate residual_gaussian_sequence() function
                self.residual_gaussian_sequence = self._simulate_lower_bound
        else:
            print('Residual approximation method is set to lower bounding method.')
            z1 = self.gig_process.cornerpoint()
            C_gamma_A = z1/(2*np.pi*self.abs_lam)
            beta_gamma_A = 0.5*self.gamma**2 + (self.abs_lam/(1+self.abs_lam))*(z1**2)/(2*self.delta**2)
            self.lb_gamma_process = GammaProcess(beta_gamma_A, C_gamma_A)
            beta_0 = 1.95 # This parameter value can be optimised further in the future...
            C_TS_A = (self.delta*np.sqrt(np.e)*np.sqrt(beta_0-1))/(np.pi*beta_0)
            beta_TS_A = 0.5*self.gamma**2 + (beta_0*z1**2)/(2*self.delta**2)
            self.lb_tempered_stable_process = TemperedStableProcess(0.5, beta_TS_A, C_TS_A)
            self.residual_gaussian_sequence = self._simulate_lower_bound

    def simulate_jumps(self):
        # Simulate the subordinator process (GIG):
        x_series, truncation_level = self.gig_process.simulate_jumps()
        # Simulate the variance-mean mixture:
        y_series = self.beta*x_series + np.sqrt(x_series)*np.random.randn(x_series.size)
        # Residual approximation:
        residual_gaussians = self.residual_gaussian_sequence(truncation_level_gamma=truncation_level, truncation_level_TS=truncation_level, size=x_series.size)
        return y_series + residual_gaussians


    def probability_density(self, x, mu=0.0):
        if (self.lam < 0) and (self.gamma == 0):
            if self.beta:
                return ((np.sqrt(2)*np.exp(self.beta*(x-mu)))/(np.sqrt(np.pi)*np.abs(self.beta)**(self.lam-0.5)))*((self.delta**2+(x-mu)**2)**((self.lam-0.5)/2))/((self.delta**(2*self.lam))*(2**(-self.lam))*gammafnc(-self.lam))*kv(self.lam-0.5, np.abs(self.beta)*np.sqrt(self.delta**2 + (x-mu)**2))
            else:
                return gammafnc(-self.lam+0.5)/(np.sqrt(np.pi*self.delta**2)*gammafnc(-self.lam))*np.power(1+((x-mu)**2)/(self.delta**2), self.lam-0.5)
        else:
            def a(lam, alpha, beta, delta):
                return ((alpha**2-beta**2)**(lam/2))/(np.sqrt(2*np.pi)*(alpha**(lam-0.5))*(delta**lam)*kv(lam, delta*np.sqrt(alpha**2-beta**2)))
            alpha = np.sqrt(self.gamma**2 + self.beta**2)
            return a(self.lam, alpha, self.beta, self.delta)*((self.delta**2+(x-mu)**2)**((self.lam-0.5)/2))*kv(self.lam-0.5, alpha*np.sqrt(self.delta**2+(x-mu)**2))*np.exp(self.beta*(x-mu))

    def unit_expected_value(self):
        return (self.delta * self.beta * kv(self.lam+1, self.delta*self.gamma)) / (self.gamma * kv(self.lam, self.delta*self.gamma))

    def unit_variance(self):
        return ((self.delta * kv(self.lam+1, self.delta*self.gamma)) / (self.gamma * kv(self.lam, self.delta*self.gamma)) 
            + ((self.beta**2 * self.delta**2)/self.gamma**2) * ( (kv(self.lam+2, self.delta*self.gamma) / kv(self.lam, self.delta*self.gamma))
            -  kv(self.lam+1, self.delta*self.gamma)**2 / kv(self.lam, self.delta*self.gamma)**2 )
        )

