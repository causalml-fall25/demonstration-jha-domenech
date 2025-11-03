import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from scipy.special import gammainc
import warnings


def calculate_mahalanobis_distance(X, W):
    """
    M = n * p_w * (1-p_w) * (X̄_T - X̄_C)' Σ⁻¹ (X̄_T - X̄_C)
    """
    n = len(W)
    n_treat = int(np.sum(W))
    p_w = n_treat / n

    X_treat_mean = X[W == 1].mean(axis=0)
    X_control_mean = X[W == 0].mean(axis=0)
    diff = X_treat_mean - X_control_mean

    Sigma = np.cov(X.T)
    Sigma_inv = np.linalg.pinv(Sigma)

    M = n * p_w * (1 - p_w) * diff.T @ Sigma_inv @ diff
    return M


def rerandomize(X, n_treat, p_a, max_attempts=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n, k = X.shape
    threshold = chi2.ppf(p_a, df=k)

    for attempt in range(1, max_attempts + 1):
        W = np.zeros(n)
        treat_indices = np.random.choice(n, size=n_treat, replace=False)
        W[treat_indices] = 1

        M = calculate_mahalanobis_distance(X, W)

        if M <= threshold:
            return W, M, attempt

    raise ValueError(f"Could not find acceptable randomization in {max_attempts} attempts")


def calculate_v_a(k, p_a):
    """
    v_a = P(χ²_{k+2} ≤ a) / P(χ²_k ≤ a)
    """
    a = chi2.ppf(p_a, df=k)
    numerator = chi2.cdf(a, df=k+2)
    denominator = p_a
    return numerator / denominator if denominator > 0 else 1.0


def plot_rerandomization_limitations(n=200, k=8, n_sims=500):
    """
    Comprehensive visualization of rerandomization limitations.
    """
    np.random.seed(42)
    
    # Generate data with nonlinear relationships
    X_nonlin = np.random.randn(n, k)
    X_nonlin = (X_nonlin - X_nonlin.mean(axis=0)) / X_nonlin.std(axis=0)
    
    # Outcome depends on linear AND quadratic terms
    beta_linear = np.array([1.5, 1.2, 0, 0, 0, 0, 0, 0])
    beta_quadratic = np.array([0.8, 0.6, 0, 0, 0, 0, 0, 0])
    
    y_nonlin = (X_nonlin @ beta_linear + 
                (X_nonlin**2) @ beta_quadratic + 
                np.random.randn(n) * 2)
    
    # Rerandomization only balances LINEAR terms
    n_treat = n // 2
    W_rerand_lin, _, _ = rerandomize(X_nonlin, n_treat, p_a=0.1, seed=42)
    
    # Calculate balance for linear and quadratic terms
    terms_to_plot = ['X₁', 'X₁²', 'X₂', 'X₂²']
    diffs = []
    for i, term in enumerate(terms_to_plot):
        covar_idx = i // 2  # 0 for X1, 1 for X2
        if '²' in term:
            diff = abs((X_nonlin[W_rerand_lin == 1, covar_idx]**2).mean() -
                       (X_nonlin[W_rerand_lin == 0, covar_idx]**2).mean())
        else:
            diff = abs(X_nonlin[W_rerand_lin == 1, covar_idx].mean() -
                       X_nonlin[W_rerand_lin == 0, covar_idx].mean())
        diffs.append(diff)
    

    # Measured covariates
    X_measured = np.random.randn(n, 4)
    X_measured = (X_measured - X_measured.mean(axis=0)) / X_measured.std(axis=0)
    
    # Hidden confounders
    X_hidden = np.random.randn(n, 3)
    X_hidden = (X_hidden - X_hidden.mean(axis=0)) / X_hidden.std(axis=0)
    
    # Outcome depends on BOTH
    beta_measured = np.array([0.3, 0.3, 0.2, 0.2])
    beta_hidden = np.array([1.5, 1.2, 1.0])
    
    signal_measured = X_measured @ beta_measured
    signal_hidden = X_hidden @ beta_hidden
    noise = np.random.randn(n) * 1.5
    
    y_hidden = signal_measured + signal_hidden + noise
    
    # Calculate R² for each component
    total_var = np.var(y_hidden)
    R2_measured = np.var(signal_measured) / total_var
    R2_hidden = np.var(signal_hidden) / total_var
    R2_noise = np.var(noise) / total_var
    
  
    X_trunc = np.random.randn(n, k)
    X_trunc = (X_trunc - X_trunc.mean(axis=0)) / X_trunc.std(axis=0)
    
    # Generate outcomes with moderate effect
    beta_trunc = np.random.uniform(0.5, 1.5, k)
    y_control_trunc = X_trunc @ beta_trunc + np.random.randn(n) * 2
    true_effect = 5.0
    
    estimates_pure = []
    estimates_moderate = []
    estimates_extreme = []
    
    for sim in range(n_sims):
        y_treat_trunc = y_control_trunc + true_effect
        
        # Pure randomization
        W_pure = np.zeros(n)
        W_pure[np.random.choice(n, n_treat, replace=False)] = 1
        y_pure = y_treat_trunc * W_pure + y_control_trunc * (1 - W_pure)
        estimates_pure.append(y_pure[W_pure == 1].mean() - y_pure[W_pure == 0].mean())
        
        # Moderate rerandomization
        W_mod, _, _ = rerandomize(X_trunc, n_treat, p_a=0.20, seed=sim)
        y_mod = y_treat_trunc * W_mod + y_control_trunc * (1 - W_mod)
        estimates_moderate.append(y_mod[W_mod == 1].mean() - y_mod[W_mod == 0].mean())
        
        # Extreme rerandomization
        W_ext, _, _ = rerandomize(X_trunc, n_treat, p_a=0.01, seed=sim*10)
        y_ext = y_treat_trunc * W_ext + y_control_trunc * (1 - W_ext)
        estimates_extreme.append(y_ext[W_ext == 1].mean() - y_ext[W_ext == 0].mean())
    
    estimates_pure = np.array(estimates_pure)
    estimates_moderate = np.array(estimates_moderate)
    estimates_extreme = np.array(estimates_extreme)
    
    X_perverse = np.random.randn(n, k)
    X_perverse = (X_perverse - X_perverse.mean(axis=0)) / X_perverse.std(axis=0)
    
    # Outcomes INDEPENDENT of covariates
    y_control_perverse = 50 + np.random.randn(n) * 5
    y_treat_perverse = y_control_perverse + true_effect
    
    coverage_perverse = []
    p_a_perverse = [0.01, 0.1, 0.3, 1.0]
    
    for p_a in p_a_perverse:
        coverages = []
        for sim in range(n_sims):
            if p_a == 1.0:
                W = np.zeros(n)
                W[np.random.choice(n, n_treat, replace=False)] = 1
            else:
                W, _, _ = rerandomize(X_perverse, n_treat, p_a, seed=sim*100)
            
            y = y_treat_perverse * W + y_control_perverse * (1 - W)
            tau_hat = y[W == 1].mean() - y[W == 0].mean()
            
            if p_a < 1.0:
                v_a = calculate_v_a(k, p_a)
                var_t = np.var(y[W == 1], ddof=1)
                var_c = np.var(y[W == 0], ddof=1)
                se = np.sqrt(v_a * (var_t / n_treat + var_c / (n - n_treat)))
            else:
                var_t = np.var(y[W == 1], ddof=1)
                var_c = np.var(y[W == 0], ddof=1)
                se = np.sqrt(var_t / n_treat + var_c / (n - n_treat))
            
            ci_lower = tau_hat - 1.96 * se
            ci_upper = tau_hat + 1.96 * se
            coverages.append(int(ci_lower <= true_effect <= ci_upper))
        
        coverage_perverse.append(np.mean(coverages) * 100)
    
  
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Nonlinearity
    ax1 = fig.add_subplot(gs[0, 0])
    colors_nonlin = ['#3498db' if '²' not in t else '#e74c3c' for t in terms_to_plot]
    bars1 = ax1.bar(range(len(terms_to_plot)), diffs, color=colors_nonlin,
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.axhline(0.15, color='#e67e22', linestyle='--', linewidth=2.5, 
                label='Acceptable imbalance', zorder=0)
    ax1.set_xticks(range(len(terms_to_plot)))
    ax1.set_xticklabels(terms_to_plot, fontsize=10)
    ax1.set_ylabel('|Mean Difference|', fontsize=10, fontweight='bold')
    ax1.set_title('(A) Nonlinearity Problem\nRerandomization Balances Linear Terms Only', 
                  fontweight='bold', fontsize=13, loc='left', pad=15)
    ax1.legend(frameon=True, loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars1, diffs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Panel B: Hidden Variables
    ax2 = fig.add_subplot(gs[0, 1])
    components = ['Measured\nCovariates\n(Balanced)', 'Hidden\nVariables\n(Unbalanced)', 'Noise']
    R2_values = [R2_measured, R2_hidden, R2_noise]
    colors_hidden = ['#27ae60', '#e74c3c', '#95a5a6']
    
    bars2 = ax2.bar(range(3), R2_values, color=colors_hidden, 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(components, fontsize=10)
    ax2.set_ylabel('Proportion of Variance Explained (R²)', fontsize=10, fontweight='bold')
    ax2.set_ylim([0, max(R2_values) * 1.25])
    ax2.set_title('(B) Hidden Confounders\nRerandomization Cannot Balance Unmeasured Variables', 
                  fontweight='bold', fontsize=13, loc='left', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars2, R2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Panel C: Truncation Bias
    ax3 = fig.add_subplot(gs[1, 0])
    methods_trunc = ['Pure\nRandomization\n(p_a=1.0)', 
                     'Moderate\nRerandomization\n(p_a=0.20)', 
                     'Extreme\nRerandomization\n(p_a=0.01)']
    biases = [
        abs(estimates_pure.mean() - true_effect),
        abs(estimates_moderate.mean() - true_effect),
        abs(estimates_extreme.mean() - true_effect)
    ]
    colors_trunc = ['#27ae60', '#f39c12', '#e74c3c']
    
    bars3 = ax3.bar(range(3), biases, color=colors_trunc, 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(methods_trunc, fontsize=10)
    ax3.set_ylabel('|Bias| in Treatment Effect', fontsize=10, fontweight='bold')
    ax3.set_title('(C) Truncation Bias\nExtreme Rerandomization Can Introduce Bias', 
                  fontweight='bold', fontsize=13, loc='left', pad=15)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars3, biases):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Panel D: Perverse Covariates
    ax4 = fig.add_subplot(gs[1, 1])
    p_a_labels = ['p_a=0.01\n(Extreme)', 'p_a=0.10\n(Strong)', 
                  'p_a=0.30\n(Moderate)', 'p_a=1.0\n(Pure Rand.)']
    colors_perverse = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60']
    
    bars4 = ax4.bar(range(4), coverage_perverse, color=colors_perverse,
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.axhline(95, color='#3498db', linestyle='--', linewidth=2.5, 
                label='Nominal 95% coverage', zorder=0)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(p_a_labels, fontsize=10)
    ax4.set_ylabel('Coverage Rate (%)', fontsize=10, fontweight='bold')
    ax4.set_ylim([0, 105])
    ax4.set_title('(D) Perverse Covariates\nRerandomization Fails When X ⊥ Y', 
                  fontweight='bold', fontsize=13, loc='left', pad=15)
    ax4.legend(frameon=True, loc='lower right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars4, coverage_perverse):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle('Four Key Limitations of Rerandomization', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('rerandomization_limitations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("KEY TAKEAWAYS: When Rerandomization Fails")
    print("="*80)
    print(f"\n(A) NONLINEARITY:")
    print(f"    • Linear terms balanced: X₁ diff = {diffs[0]:.3f}, X₂ diff = {diffs[2]:.3f}")
    print(f"    • Quadratic terms imbalanced: X₁² diff = {diffs[1]:.3f}, X₂² diff = {diffs[3]:.3f}")
    print(f"    → Solution: Include nonlinear terms in covariate matrix")
    
    print(f"\n(B) HIDDEN CONFOUNDERS:")
    print(f"    • Measured covariates explain: {R2_measured*100:.1f}% of variance")
    print(f"    • Hidden variables explain: {R2_hidden*100:.1f}% of variance")
    print(f"    → Solution: Measure all important prognostic variables pre-experiment")
    
    print(f"\n(C) TRUNCATION BIAS:")
    print(f"    • Pure randomization bias: {biases[0]:.3f}")
    print(f"    • Moderate (p_a=0.20) bias: {biases[1]:.3f}")
    print(f"    • Extreme (p_a=0.01) bias: {biases[2]:.3f}")
    print(f"    → Solution: Use moderate p_a (0.10-0.30) to balance precision & validity")
    
    print(f"\n(D) PERVERSE COVARIATES:")
    print(f"    • p_a=0.01 coverage: {coverage_perverse[0]:.1f}% (should be 95%!)")
    print(f"    • p_a=1.0 coverage: {coverage_perverse[3]:.1f}% ✓")
    print(f"    → Solution: Only include prognostic covariates (X correlated with Y)")
    print("="*80)
