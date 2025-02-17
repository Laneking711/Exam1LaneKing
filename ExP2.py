##############################################################################
# region imports
import math
import random
import NumMet as nm  # Make sure NumMet.py is available for Simpson integration
#endregion
##############################################################################
#Chatgpt was consulted to assist in the production of this code
#Previous Homework 3 used in some places
#Professor smays homework 3 solution used as refference 
##############################################################################
# region: truncated lognormal + mean/variance
def truncated_lognormal_sample(mu, sigma, d_min, d_max, size=1):
    """
    Generate 'size' samples from a log-normal distribution truncated
    to the interval [d_min, d_max].

    Args:
        mu (float): Mean of ln(D).
        sigma (float): Std. dev. of ln(D).
        d_min (float): Lower bound of truncation for diameters.
        d_max (float): Upper bound of truncation for diameters.
        size (int): Number of samples to generate.

    Returns:
        list of float: The truncated lognormal samples.
    """
    samples = []
    while len(samples) < size:
        z = random.gauss(mu, sigma)  # z ~ N(mu, sigma)
        d = math.exp(z)              # Convert to log-normal
        if d_min <= d <= d_max:
            samples.append(d)
    return samples

def mean_variance(data):
    """
    Calculate the mean and (sample) variance of a list of numbers.

    Args:
        data (list of float): List of numerical values.

    Returns:
        (mean_, var_): Tuple containing the sample mean and sample variance.
                       If data is empty, returns (None, None).
    """
    n = len(data)
    if n == 0:
        return (None, None)
    mean_ = sum(data) / n
    var_ = sum((x - mean_) ** 2 for x in data) / (n - 1)  # sample variance
    return mean_, var_
#endregion
##############################################################################

##############################################################################
# region: t-distribution probability code
def gamma_function(alpha):
    """
    Simple wrapper for math.gamma(), which computes Γ(alpha).

    Args:
        alpha (float): Input value for the Gamma function.

    Returns:
        float: Γ(alpha)
    """
    return math.gamma(alpha)

def km(m):
    """
    Compute the normalization constant K_m for the Student t-distribution
    with 'm' degrees of freedom.

    K_m = Γ((m+1)/2) / [ sqrt(m * math.pi) * Γ(m/2) ]

    Args:
        m (float): Degrees of freedom (often a positive float in Welch's test).

    Returns:
        float: The constant K_m for the given m.
    """
    return gamma_function(0.5 * m + 0.5) / (
           math.sqrt(m * math.pi) * gamma_function(0.5 * m)
    )

def t_pdf(x, m):
    """
    Compute the PDF of the Student t-distribution at x, for 'm' degrees of freedom.

    PDF(x; m) = K_m * [1 + (x^2 / m)]^(-(m+1)/2)

    Args:
        x (float): The point at which we want the PDF value.
        m (float): Degrees of freedom.

    Returns:
        float: t-distribution PDF value at x.
    """
    base = 1.0 + (x * x) / m
    exponent = -(m + 1) / 2
    return km(m) * (base ** exponent)

def normal_cdf(z):
    """
    Standard normal CDF using the error function:
        Φ(z) = 0.5 * [1 + erf(z / sqrt(2))]

    Args:
        z (float): The z-score at which to evaluate the CDF.

    Returns:
        float: The cumulative probability from -∞ up to z for a standard normal.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def t_cdf(t_value, m):
    """
    Numerically compute the CDF of the t-distribution at 't_value':
        F(t_value; m) = ∫ PDF(x;m) dx from -∞ to t_value

    To prevent overflow:
      - If m >= 100, approximate using the standard normal cdf.
      - Otherwise clamp integration to [-30, +30], because beyond that
        the tail contribution is negligible.

    Args:
        t_value (float): The t statistic or x-value at which to compute CDF.
        m (float): Degrees of freedom (can be fractional in Welch's test).

    Returns:
        float: The cumulative probability F(t_value; m).
    """
    # For large df, use normal approximation
    if m >= 100:
        return normal_cdf(t_value)

    # Clamp integration domain to avoid huge tails
    if t_value > 30.0:
        return 1.0
    if t_value < -30.0:
        return 0.0

    lower_bound = -30.0
    upper_bound = t_value

    def pdf_wrapper(args):
        x, dof = args[0], args[1]
        return t_pdf(x, dof)

    # Integrate the PDF from -30 up to t_value
    cdf_val = nm.Simpson(pdf_wrapper, (m, 0, lower_bound, upper_bound), 2000)
    return cdf_val

def t_critical_value(alpha, df, tail="two-sided"):
    """
    Returns the critical value (c_crit) for a t-distribution with 'df' degrees of freedom,
    at significance level alpha, using a simple numerical search that inverts t_cdf.

    Args:
        alpha (float): Significance level (e.g. 0.05).
        df (float): Degrees of freedom for the t-distribution.
        tail (str): Either "two-sided" or "one-sided".
                    - "two-sided": we find c where P(T <= c) = 1 - alpha/2
                    - "one-sided": we find c where P(T <= c) = 1 - alpha

    Returns:
        float: The critical t-value c for which the upper tail area is alpha/2 (two-sided)
               or alpha (one-sided).
    """
    # Decide target probability
    if tail.lower() == "two-sided":
        p_target = 1.0 - alpha/2.0
    else:  # "one-sided"
        p_target = 1.0 - alpha

    # We'll do a simple binary search in [0, 30] for the positive c
    # If you need the negative bound for a symmetric two-sided test, you'd just do -c
    lower, upper = 0.0, 30.0
    for _ in range(100):  # 100 iterations is plenty for typical double-precision
        mid = 0.5 * (lower + upper)
        cdf_mid = t_cdf(mid, df)
        if cdf_mid < p_target:
            lower = mid
        else:
            upper = mid
    c_crit = 0.5 * (lower + upper)
    return c_crit
#endregion
##############################################################################

##############################################################################
# region: Welch's t-test (one-sided)
def welch_ttest_onesided(sampleA, sampleB, alpha=0.05):
    """
    Perform a one-sided Welch's t-test to check if mean(B) < mean(A).
    Hypotheses:
        H0: mu_B >= mu_A
        H1: mu_B < mu_A

    Returns:
        (t_stat, df_eff, p_value, conclusion_string)

      - t_stat: the computed Welch's t statistic
      - df_eff: the effective degrees of freedom via Welch-Satterthwaite
      - p_value: one-sided p-value = P(T <= t_stat) under t-dist with df_eff
      - conclusion_string: textual interpretation (Reject/Fail to reject H0).
    """
    nA = len(sampleA)
    nB = len(sampleB)

    # Compute sample means & variances
    mA, varA = mean_variance(sampleA)
    mB, varB = mean_variance(sampleB)

    # If variance is extremely small, set a floor to avoid division by zero
    if varA < 1e-12:
        varA = 1e-12
    if varB < 1e-12:
        varB = 1e-12

    # Welch's t-statistic
    numerator = (mB - mA)
    denom = math.sqrt(varB / nB + varA / nA)
    t_stat = numerator / denom

    # Welch–Satterthwaite approximation for degrees of freedom
    df_num = (varB / nB + varA / nA) ** 2
    df_den = (varB ** 2 / (nB ** 2 * (nB - 1))) + (varA ** 2 / (nA ** 2 * (nA - 1)))
    if df_den <= 1e-18:
        df_eff = (nA + nB - 2)
    else:
        df_eff = df_num / df_den

    # One-sided p-value = P(T <= t_stat) for T ~ t_dist(df_eff)
    cdf_val = t_cdf(t_stat, df_eff)
    p_value = cdf_val

    # Conclusion
    if p_value < alpha:
        conclusion = (
            f"Reject H0 at alpha={alpha}. Supplier B's mean diameter is "
            f"statistically smaller (p={p_value:.4g})."
        )
    else:
        conclusion = (
            f"Fail to reject H0 at alpha={alpha}. No significant difference "
            f"in favor of B being smaller (p={p_value:.4g})."
        )

    return t_stat, df_eff, p_value, conclusion
#endregion
##############################################################################

##############################################################################
# region: main program
def main():
    """
    Main driver function which orchestrates the entire analysis:

    1. Prompt the user for feedstock ln(D) mean and std. dev.
       (common to both company A and B).
    2. Prompt for company A & B aperture sizes (large & small),
       the number of samples, and items per sample.
    3. Generate truncated lognormal data for each company based on the inputs.
    4. Perform a one-sided Welch's t-test (B < A) at alpha=0.05.
    5. Compute the critical t-value c_crit for one-sided alpha=0.05 and df_eff.
    6. Print out the results, including sample means, variance,
       the test-statistic t, degrees of freedom, p-value, and c_crit (the
       critical value from the distribution).
    """
    # Ask user for the lognormal parameters that define the "pre-sieved" rock distribution
    print("Enter parameters for the feedstock that is common to both company A and company B.")
    try:
        mu_str = input("Mean of ln(D) for the pre-sieved rocks? (e.g. 0.693 ~ ln(2)): ").strip()
        mu_feed = float(mu_str) if mu_str else 0.693

        sigma_str = input("Std. dev. of ln(D) for the pre-sieved rocks? [default=1.0]: ").strip()
        sigma_feed = float(sigma_str) if sigma_str else 1.0
    except ValueError:
        print("Invalid input. Using defaults: mu=0.693, sigma=1.0")
        mu_feed, sigma_feed = 0.693, 1.0

    # Ask the user for Company A's aperture sizes and sampling info
    print("\nEnter parameters for company A sieving and sampling operations:")
    try:
        laA_str = input("  Large aperture size? [default=1.0]: ").strip()
        largeA = float(laA_str) if laA_str else 1.0

        saA_str = input("  Small aperture size? [default=0.375]: ").strip()
        smallA = float(saA_str) if saA_str else 0.375

        nSamplesA_str = input("  How many samples? [default=11]: ").strip()
        nSamplesA = int(nSamplesA_str) if nSamplesA_str else 11

        nItemsA_str = input("  How many items in each sample? [default=100]: ").strip()
        nItemsA = int(nItemsA_str) if nItemsA_str else 100

    except ValueError:
        print("Invalid input. Using: large=1.0, small=0.375, 11 samples, 100 items each.")
        largeA, smallA, nSamplesA, nItemsA = 1.0, 0.375, 11, 100

    # Ask the user for Company B's aperture sizes and sampling info
    print("\nEnter parameters for company B sieving and sampling operations:")
    try:
        laB_str = input("  Large aperture size? [default=0.875]: ").strip()
        largeB = float(laB_str) if laB_str else 0.875

        saB_str = input("  Small aperture size? [default=0.375]: ").strip()
        smallB = float(saB_str) if saB_str else 0.375

        nSamplesB_str = input("  How many samples? [default=11]: ").strip()
        nSamplesB = int(nSamplesB_str) if nSamplesB_str else 11

        nItemsB_str = input("  How many items in each sample? [default=100]: ").strip()
        nItemsB = int(nItemsB_str) if nItemsB_str else 100

    except ValueError:
        print("Invalid input. Using: large=0.875, small=0.375, 11 samples, 100 items each.")
        largeB, smallB, nSamplesB, nItemsB = 0.875, 0.375, 11, 100

    # Compute the total number of data points for each supplier
    bigSampleA_size = nSamplesA * nItemsA
    bigSampleB_size = nSamplesB * nItemsB

    # Generate truncated lognormal samples for Company A and B
    sampleA = truncated_lognormal_sample(mu_feed, sigma_feed, smallA, largeA, bigSampleA_size)
    sampleB = truncated_lognormal_sample(mu_feed, sigma_feed, smallB, largeB, bigSampleB_size)

    # Perform a one-sided Welch's t-test at alpha=0.05 to check if B's mean < A's mean
    alpha = 0.05
    t_stat, df_eff, p_val, conclusion = welch_ttest_onesided(sampleA, sampleB, alpha=alpha)

    # Also compute the one-sided critical t-value for the same alpha, df
    # so we can compare t_stat with that critical cutoff.
    c_crit = t_critical_value(alpha, df_eff, tail="one-sided")

    # Calculate sample means and variances for the final report
    mA, vA = mean_variance(sampleA)
    mB, vB = mean_variance(sampleB)

    # Display all relevant results to the user
    print(f"\nSummary of parameters for feedstock: ln(D) mean={mu_feed:.3f}, stdev={sigma_feed:.3f}")
    print("--------------------------------------------------------------------")
    print(f"Company A: large={largeA}, small={smallA}, total n={bigSampleA_size}")
    print(f"Company B: large={largeB}, small={smallB}, total n={bigSampleB_size}")

    print("\nResults:")
    print(f"  Supplier A sample mean={mA:.3f}, var={vA:.3f}, n={bigSampleA_size}")
    print(f"  Supplier B sample mean={mB:.3f}, var={vB:.3f}, n={bigSampleB_size}")
    print(f"  Computed t-statistic = {t_stat:.3f}")
    print(f"  C- (alpha={alpha}) = {c_crit:.3f}")

    print(f"\n{conclusion}")
    print("\nDone.\n")
#endregion
##############################################################################

if __name__ == "__main__":
    main()
