import torch
import torch.distributions as dist
from torch import nn

import math

class HMM_ensemble(torch.nn.Module):
    """
    Hidden Markov Model with discrete observations.
    """
    def __init__(self, n_state_list, m_dimensions, max_iterations = 100,
                tolerance = 0.1, verbose = True, lambda_initate_list = None,
                certainty = 0.01, confidence_rate = 1.5, entropy_threshold = 0.1,
                ensemble_assignment = None, high_prob = 0.5, entropy_break = 5):
        super(HMM_ensemble, self).__init__()
        self.n_state_list = n_state_list  # number of states
        self.T_max = None # Max time step
        self.m_dimensions = m_dimensions
        self.num_latent_variables = len(self.n_state_list)
        
        self.max_iterations = max_iterations 
        self.tolerance = tolerance
        self.entropy_break = entropy_break # Number of iterations after zero entropy

        self.epsilon = 1e-32 #To ensure positive proabilities

        self.verbose = verbose
        
        self.test_sum_list = []

        # A_mat
        self.log_transition_matrix_list = nn.ParameterList([
            nn.Parameter(torch.log(torch.nn.functional.softmax(torch.randn(N, N), dim=0)))
            for N in n_state_list
        ])

        # b(x)
        self.lambda_list = nn.ParameterList([
            nn.Parameter(torch.exp(torch.randn(N, m_dimensions)))
            for N in n_state_list
        ])

        # pi
        self.log_state_priors_list = nn.ParameterList([
            nn.Parameter(torch.log_softmax(torch.randn(N), dim=0))
            for N in n_state_list
        ])

        # Z
        if ensemble_assignment is not None:
            # Ensure high_prob is not too high to allow for other probabilities
            assert high_prob < 1.0, "high_prob must be less than 1.0"

            # Calculate the probability to be distributed among other variables
            low_prob = (1.0 - high_prob) / (self.num_latent_variables - 1)

            # Initialize ensemble_probabilities
            self.ensemble_probabilities = torch.full((self.m_dimensions, self.num_latent_variables), low_prob).float()

            # Assign the higher probability according to ensemble_assignment
            for i in range(self.m_dimensions):
                if self.ensemble_assignment[i] == -1:
                    self.ensemble_probabilities[i, :] = 1 / self.num_latent_variables 
                else:
                    self.ensemble_probabilities[i, self.ensemble_assignment[i]] = high_prob

        else:
            self.ensemble_probabilities = torch.ones(self.m_dimensions, self.num_latent_variables).float()/self.num_latent_variables
            self.ensemble_assignment = torch.argmax(self.ensemble_probabilities, dim = 1)
        
        self.active_ensembles = torch.ones(self.num_latent_variables).int()
        ensemble_probabilities_log_sum = torch.log(torch.sum(self.ensemble_probabilities, dim = 0) + self.epsilon)
        self.ensemble_priors = torch.softmax(ensemble_probabilities_log_sum, dim = 0)

        self.certainty = certainty
        self.confidence_rate = confidence_rate
        self.entropy = mean_entropy(self.ensemble_probabilities)
        self.entropy_threshold = entropy_threshold
        

        # use the GPU
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.cuda()

    def get_transition_matrix(self):
        return [torch.exp(log_transition_matrix) for log_transition_matrix in self.log_transition_matrix_list]

    def get_state_priors(self):
        return [torch.exp(log_state_prior) for log_state_prior in self.log_state_priors_list]
    
    def emission_model(self, x):
        """
        x: LongTensor of shape (T_max, m_dimensions)

        returns:
        log_probabilities_list: List of length self.num_latent_variables,
            each with LongTensor of shape (T_max, n, m), where n is the number of states for that laten.

        Get observation log probabilities
        """
        # Compute Poisson log probabilities for each lambda in parallel
        tot_log_probabilities_list = []

        for i in range(self.num_latent_variables):
            lambdas = self.lambda_list[i]
            poisson_dist = dist.Poisson(lambdas)
            tot_log_probabilities = torch.zeros(x.shape[0], self.n_state_list[i], self.m_dimensions)

            for t in range(x.shape[0]):
                log_prob = poisson_dist.log_prob(x[t,:])
                tot_log_probabilities[t,:,:] = log_prob
            tot_log_probabilities_list.append(tot_log_probabilities)

        return tot_log_probabilities_list
            
                    
    def log_alpha_calc(self):
            """
            Calculate log-alpha values using provided model parameters.

            Returns:
                log_alpha: List of FloatTensors, each of shape (T_max, n_states, m_dimensions)
            """

            log_alpha_list = [
                torch.zeros(self.T_max, n, self.m_dimensions).float().cuda() if self.is_cuda else torch.zeros(self.T_max, n, self.m_dimensions).float()
                for n in self.n_state_list
            ]

            for i, (log_emission_matrix, log_state_priors, log_transition_matrix) in enumerate(zip(self.log_emission_matrix_list, self.log_state_priors_list, self.log_transition_matrix_list)):
                # Create the initial alpha values using the state priors and the first emission probabilities
                log_alpha_i = log_alpha_list[i]
                log_state_priors_expanded = log_state_priors.unsqueeze(0).expand(self.m_dimensions, -1).t()
                log_alpha_i[0] = log_emission_matrix[0] + log_state_priors_expanded

                # Update alpha values over time
                for t in range(1, self.T_max):
                    log_alpha_i[t] = log_emission_matrix[t] + log_domain_matmul(log_transition_matrix, log_alpha_i[t-1])

                log_alpha_list[i] = log_alpha_i

            return log_alpha_list
            
        
    def log_beta_calc(self):
        assert self.log_emission_matrix_list is not None, "No emission matrix"
        
        # With x dimension
        log_beta_list = [
            torch.zeros(self.T_max, n, self.m_dimensions).float().cuda() if self.is_cuda else torch.zeros(self.T_max, n, self.m_dimensions).float()
            for n in self.n_state_list
        ]  
        for i in range(self.num_latent_variables):
            log_beta_i = log_beta_list[i]
            log_emission_matrix = self.log_emission_matrix_list[i]
            log_transition_matrix = self.log_transition_matrix_list[i]
        
            for t in range(self.T_max - 2, -1, -1):
                log_beta_i[t, :, :] = torch.logsumexp(
                    log_beta_i[t + 1, :, :] +  # Broadcasting
                    log_transition_matrix[:, :, None] + 
                    log_emission_matrix[t + 1, :, :], 
                    dim=1   
                )

            log_beta_list[i] = log_beta_i
        
        return log_beta_list
    
    def forward(self, x):
        """
        Compute log p(x) in a vectorized and efficient manner.
        x: IntTensor of shape (T_max, m_dimensions)
        """
        self.T_max = x.shape[0]
        self.log_emission_matrix_list = self.emission_model(x)
        log_alpha_list = self.log_alpha_calc()

        # Initialize total log probability tensor directly without intermediate list storage
        total_log_prob = torch.zeros_like(self.ensemble_probabilities[:, 0])

        # Compute log probabilities across all latent variables directly accumulating the results
        for i, log_alpha in enumerate(log_alpha_list):
            log_prob_x = log_alpha[self.T_max-1,:,:].logsumexp(dim=0)  # sum over states
            total_log_prob += torch.log(self.ensemble_probabilities[:, i] + self.epsilon) + log_prob_x  # Accumulate directly in log space

        # Summing up the total log probabilities once, after all contributions have been accumulated
        total_log_prob = total_log_prob.logsumexp(dim=0).sum()  # sum over ensemble probabilities
        
        return total_log_prob

    
    def fit(self, x):
        """ Estimates optimal transition matrix and lambdas given the data x.

        Args:
            x (torch): T_max x m_dimensions
            log_alpha (torch) : T_max x N
            log_beta (torch) : T_max x N
        """
        entropy_break_count = 0
        self.T_max = x.shape[0]
        prev_log_likelihood = float('-inf')
        log_x = torch.log(x + self.epsilon)
        
        for iteration in range(self.max_iterations):
            # Get emission matrix
            self.log_emission_matrix_list = self.emission_model(x)
            
            # E step
            ## Calculate log_alpha
            log_alpha_list = self.log_alpha_calc()
            
            ## Caculcate log_beta
            log_beta_list = self.log_beta_calc()
                        
                        
            ## Calculate Ensemble Proabilities
            for m in range(self.m_dimensions):
                latent_log_like = torch.zeros(self.num_latent_variables).float()
                for i in range(self.num_latent_variables):
                    if self.active_ensembles[i] == 0:
                        latent_log_like[i] = torch.log(self.epsilon)
                    else:
                        log_alpha = log_alpha_list[i]
                        latent_log_like[i] = log_alpha[self.T_max -1,:, m].logsumexp(dim=0) 

                self.ensemble_probabilities[m,:] = torch.softmax(self.certainty * latent_log_like , dim = 0)
                if torch.sum(self.ensemble_probabilities[m,:]) == 0:
                    self.active_ensembles[i] = 0
            self.ensemble_assignment = torch.argmax(self.ensemble_probabilities, dim = 1)

            total_log_prob = torch.zeros_like(self.ensemble_probabilities[:, 0])

            # Compute log probabilities across all latent variables directly accumulating the results
            for i, log_alpha in enumerate(log_alpha_list):
                log_prob_x = log_alpha[self.T_max-1,:,:].logsumexp(dim=0)  # sum over states
                total_log_prob += torch.log(self.ensemble_probabilities[:, i] + self.epsilon) + log_prob_x  # Accumulate directly in log space

            # Summing up the total log probabilities once, after all contributions have been accumulated
            log_likelihood = total_log_prob.logsumexp(dim=0).sum()  # sum over ensemble probabilities
            
            log_likelihood_change = log_likelihood - prev_log_likelihood
            prev_log_likelihood = log_likelihood
            if self.verbose:
                if log_likelihood_change > 0:
                    print(f"{iteration + 1} {log_likelihood:.4f}  +{log_likelihood_change}")
                else:
                    print(f"{iteration + 1} {log_likelihood:.4f}  {log_likelihood_change}")
            
            if log_likelihood_change < self.tolerance and log_likelihood_change > 0:
                if self.verbose:
                    print("Converged (change in log likelihood within tolerance)")
                break
            if math.isnan(log_likelihood):
                break

            for i in range(self.num_latent_variables):
                ## Calculate log_gamma
                log_alpha = log_alpha_list[i]
                log_beta = log_beta_list[i]
                log_transition_matrix = self.log_transition_matrix_list[i]
                log_emission_matrix = self.log_emission_matrix_list[i]
                e_prob = self.ensemble_probabilities[:,i] # Dim (M)

                gamma_numerator = log_alpha + log_beta
                gamma_denominator = gamma_numerator.logsumexp(dim=1, keepdim=True)

                # Has dimension (T, N, M)
                log_gamma = gamma_numerator - gamma_denominator.expand_as(gamma_numerator)

            
                ## Calculate log_xi
                xi_sum_numerator = log_alpha[:-1,:, None,:] + log_transition_matrix[None, :, :, None ]
                xi_numerator = xi_sum_numerator + log_beta[1:,None, :, :] + log_emission_matrix[1:, None, :, : ]
                
                xi_denominator = xi_numerator.logsumexp(dim = (1,2), keepdim=True)
                
                # Has dimensions (T-1, N, N, M)
                log_xi = xi_numerator - xi_denominator

                
                # M step
                ## Update pi 
                state_numerator = torch.log(torch.matmul(torch.exp(log_gamma[0,:]),self.ensemble_probabilities[:,i])+self.epsilon) 
                gamma_tot_0 = torch.exp(log_gamma[0,:].logsumexp(dim = 0))
                state_denominator = torch.log(torch.matmul(gamma_tot_0, self.ensemble_probabilities[:,i])+ self.epsilon)

                self.log_state_priors_list[i] = state_numerator - state_denominator.expand_as(state_numerator)
                
                ## Updaten transition matrix
                log_xi_sum = log_xi.logsumexp(dim = 0)
                log_numerator = (torch.log(e_prob[None, None, :] + self.epsilon) + log_xi_sum).logsumexp(dim = 2)

                log_gamma_sum = log_gamma[0:(self.T_max-1),:].logsumexp(dim = 0)
                log_denominator = (torch.log(e_prob[None, :] + self.epsilon) + log_gamma_sum).logsumexp(dim = 1)

                self.log_transition_matrix_list[i] = log_numerator - log_denominator.view(-1, 1)
                
                # Updata lambda
                log_gamma_collapsed = torch.logsumexp(torch.log(e_prob[None, None,:] + self.epsilon) + log_gamma, dim = 2) - torch.logsumexp(torch.log(e_prob[None, None,:] + self.epsilon), dim = 2)

                lambda_numerator  = log_domain_matmul(log_gamma_collapsed.t(), log_x)
                lambda_denominator = log_gamma_collapsed.logsumexp(dim = 0)

                self.lambda_list[i] = torch.exp(lambda_numerator  - lambda_denominator.view(-1,1))
                
                self.ensemble_assignment = torch.argmax(self.ensemble_probabilities, dim = 1)
                self.certainty = self.certainty * self.confidence_rate

                ensemble_probabilities_log_sum = torch.log(torch.sum(self.ensemble_probabilities, dim = 0) + self.epsilon)
                self.ensemble_priors = torch.softmax(ensemble_probabilities_log_sum, dim = 0)
                
            
            self.entropy = mean_entropy(self.ensemble_probabilities)
            if self.verbose:
                print("Entropy:", self.entropy)
                # print("Certianty:", self.certainty)
            
            if self.verbose and iteration == self.max_iterations -1:
                print("Max itteration reached.")

            if mean_entropy(self.ensemble_probabilities) == 0:
                entropy_break_count += 1
                if entropy_break_count == self.entropy_break:
                    break
                
        
    def predict(self, x):
        """
        x: IntTensor of shape (T_max, m_dimensions)

        Find argmax_z log p(z|x)
        """
        if self.is_cuda:
            x = x.cuda()
        
        T_max = x.shape[0]
        self.m_dimensions = x.shape[1]
        z_star = torch.zeros(T_max, self.num_latent_variables).long()
        log_emission_matrix_list = self.emission_model(x)

        for i in range(self.num_latent_variables):
            n = self.n_state_list[i]
            e_assign = self.ensemble_assignment
            is_ensemble = (e_assign == i)
            if is_ensemble.sum() == 0:
                print("ensemble nr", i, "skipped.")
                continue

            log_state_priors = self.log_state_priors_list[i]
            log_emission_matrix = log_emission_matrix_list[i]
            log_transition_matrix = self.log_transition_matrix_list[i]

            log_emission_matrix_i = log_emission_matrix[:,:,is_ensemble].sum(dim = 2)

            log_delta = torch.zeros(T_max, n).float()
            psi = torch.zeros(T_max, n).long()

            log_delta[0,:] = log_emission_matrix_i[0,:] + log_state_priors
            for t in range(1, T_max):
                log_delta_expanded = log_delta[t-1,:].unsqueeze(0).expand(log_transition_matrix.size(0), -1)
                elementwise_sums = log_transition_matrix + log_delta_expanded
                max_val, argmax_val = torch.max(elementwise_sums, dim = 1)
                
                log_delta[t,:] = log_emission_matrix_i[t,:] + max_val
                psi[t,:] = argmax_val

            z_star[T_max-1,i] = log_delta[T_max -1, :].argmax()
            for t in range(T_max -2, -1,-1):
                z_star[t,i] = psi[t+1,z_star[t+1,i]]

        return z_star 
            

def mean_entropy(tensor):
    # Assume tensor is a 2D tensor where each row is a set of probabilities
    # Calculate entropy for each set of probabilities along dim 1
    entropies = -torch.sum(tensor * torch.log(tensor + 1e-16), dim=1)
    return torch.mean(entropies).item()

def log_domain_matmul(log_A, log_B, maxmul = False):
    """
    Performs log-domain matrix multiplication between log_A and log_B.

    Args:
        log_A (torch.Tensor): The logarithm of matrix A with shape (m, p).
        log_B (torch.Tensor): The logarithm of matrix B with shape (p, n).

    Returns:
        torch.Tensor: The logarithm of the matrix product of A and B with shape (m, n).
    """
    # Expand log_A to (m, 1, p) and log_B to (1, n, p) for broadcasting
    log_A_expanded = log_A.unsqueeze(1)
    log_B_expanded = log_B.transpose(0, 1).unsqueeze(0)

    # Compute the log-domain matrix multiplication
    elementwise_sums = log_A_expanded + log_B_expanded
    if maxmul:
        out1, out2 = torch.max(elementwise_sums, dim = 1)
        return out1, out2
    return torch.logsumexp(elementwise_sums, dim=2)