# psa-marketing-final


# Propensity Score Matching: Marketing Example
## A Complete Tutorial for Causal Inference

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Prerequisites](#prerequisites)
4. [Why PSM? The Confounding Problem](#why-psm)
5. [Step-by-Step Implementation](#implementation)
6. [Interpreting Results](#interpretation)
7. [Common Pitfalls](#pitfalls)
8. [Exercises](#exercises)
9. [Additional Resources](#resources)

---

## Overview {#overview}

This tutorial teaches **Propensity Score Matching (PSM)** through a practical marketing example:

**Research Question:** *Did offering a discount to customers increase their probability of making a purchase?*

**The Challenge:** Marketers didn't randomly assign discounts. They targeted customers based on characteristics like loyalty and engagement—the same factors that influence purchase behavior. This creates **confounding**, making simple comparisons biased.

**The Solution:** PSM helps us estimate causal effects from observational data by creating balanced comparison groups based on the probability of receiving treatment.

---

## Learning Objectives {#learning-objectives}

After completing this tutorial, you will be able to:

1. ✅ **Identify confounding** and explain why naive comparisons are biased
2. ✅ **Estimate propensity scores** using logistic regression
3. ✅ **Implement matching algorithms** (nearest neighbor with calipers)
4. ✅ **Assess covariate balance** using standardized mean differences
5. ✅ **Estimate treatment effects** with confidence intervals
6. ✅ **Conduct robustness checks** and interpret limitations

---

## Prerequisites {#prerequisites}

**Required Knowledge:**
- Basic statistics (means, standard deviations, hypothesis testing)
- Understanding of logistic regression
- Familiarity with Python and pandas

**Software Requirements:**
```bash
pip install numpy pandas scikit-learn statsmodels matplotlib seaborn
```

---

## Why PSM? The Confounding Problem {#why-psm}

### The Naive Approach (Wrong!)

Suppose we simply compare purchase rates:
- **Treated group** (offered discount): 35% purchased
- **Control group** (no discount): 27% purchased
- **Naive difference**: 8 percentage points

**Problem:** This comparison is **biased** because the groups differ in ways that affect purchases:

| Characteristic | Treated | Control | Impact on Purchase |
|----------------|---------|---------|-------------------|
| Loyalty Score | 0.45 | 0.32 | ⬆️ Higher loyalty → more purchases |
| Prior Engagement | 2.8 visits | 1.6 visits | ⬆️ More visits → more purchases |
| Age | 38 years | 42 years | Mixed effect |

The 8-point difference conflates:
1. **True causal effect** of the discount
2. **Selection bias** from targeting high-propensity customers

### The PSM Solution

PSM addresses this by:

1. **Modeling treatment assignment**: Estimate P(treated | customer characteristics)
2. **Creating matched pairs**: Find control customers similar to treated ones
3. **Balancing covariates**: Ensure groups look identical on observed characteristics
4. **Estimating effects**: Compare outcomes within matched pairs

**Key Insight:** If we match on propensity scores, we simulate a randomized experiment *on observed covariates*.

---

## Step-by-Step Implementation {#implementation}

### Step 1: Generate or Load Data

For this tutorial, we use synthetic data where we know the true effect (8 percentage points):

```python
# 5000 customers with characteristics
df = generate_synthetic_marketing_data(n=5000)

# Variables:
# - age, income, prior_engagement, loyalty_score, channel
# - treatment (1=discount offered, 0=no discount)
# - purchase (1=bought, 0=didn't buy)
```

**In real applications:** You would load your observational dataset here.

---

### Step 2: Exploratory Data Analysis

**Check for confounding:**

```python
# Are treated customers different?
df.groupby('treatment')[['loyalty_score', 'prior_engagement']].mean()
```

Expected output:
```
             loyalty_score  prior_engagement
treatment                                     
0                    0.32              1.62
1                    0.45              2.84
```

**Observation:** Treated customers have higher loyalty and engagement! This is confounding.

---

### Step 3: Estimate Propensity Scores

The propensity score is P(Treatment=1 | X), where X represents all covariates.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Prepare features (one-hot encode categorical variables)
df_model = pd.get_dummies(df, columns=['channel'], drop_first=True)
covariates = ['age', 'income', 'prior_engagement', 'loyalty_score', 
              'channel_sms', 'channel_social']

# Standardize continuous variables
X = df_model[covariates].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression
ps_model = LogisticRegression(C=1e6, max_iter=1000)
ps_model.fit(X_scaled, df_model['treatment'])

# Predict propensity scores
df_model['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
```

**Important Considerations:**

- **Model Choice:** Logistic regression is standard and interpretable. Alternatives include random forests or gradient boosting (be careful of overfitting).
- **Variable Selection:** Include all confounders (variables affecting both treatment and outcome). Omitting confounders causes bias.
- **Functional Form:** Consider interactions or non-linear terms if relationships are complex.

---

### Step 4: Check Overlap (Common Support)

Plot propensity score distributions:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.kdeplot(df_model[df_model.treatment==1]['propensity_score'], 
            label='Treated', fill=True)
sns.kdeplot(df_model[df_model.treatment==0]['propensity_score'], 
            label='Control', fill=True)
plt.xlabel('Propensity Score')
plt.title('Overlap Check')
plt.legend()
plt.show()
```

**What to look for:**
- ✅ **Good overlap:** Distributions overlap substantially
- ⚠️ **Poor overlap:** Little overlap means few comparable units; estimates will be unstable

**Action if overlap is poor:**
- Trim extreme propensity scores (e.g., keep 0.1 < PS < 0.9)
- Re-specify propensity model
- Accept that estimates apply only to the overlap region

---

### Step 5: Perform Matching

**1:1 Nearest Neighbor Matching with Caliper:**

```python
from sklearn.neighbors import NearestNeighbors

def nearest_neighbor_match(df, caliper=0.05):
    treated = df[df.treatment==1].copy()
    control = df[df.treatment==0].copy()
    
    # Build nearest neighbor model on propensity scores
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['propensity_score']])
    
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    
    # Apply caliper (reject matches too far apart)
    pairs = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        ps_diff = abs(treated.iloc[i]['propensity_score'] - 
                     control.iloc[idx[0]]['propensity_score'])
        if ps_diff <= caliper:
            pairs.append((i, idx[0]))
    
    return pairs

pairs = nearest_neighbor_match(df_model, caliper=0.05)
print(f"Matched {len(pairs)} pairs")
```

**Matching Parameters:**

- **Ratio:** 1:1 (one control per treated), 1:2, 1:k, or variable
- **Replacement:** With (controls can be reused) or without
- **Caliper:** Maximum allowed propensity score difference
  - Rule of thumb: 0.2 × SD(logit(propensity))
  - Tighter calipers → better balance, fewer matches

---

### Step 6: Assess Covariate Balance

**Standardized Mean Difference (SMD):**

SMD = (Mean_treated - Mean_control) / Pooled_SD

**Rule of thumb:** |SMD| < 0.1 indicates good balance

```python
def compute_smd(matched_df, covariates):
    smd = {}
    for cov in covariates:
        treated = matched_df[matched_df.role=='treated'][cov]
        control = matched_df[matched_df.role=='control'][cov]
        
        mean_diff = treated.mean() - control.mean()
        pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
        smd[cov] = mean_diff / pooled_sd
    
    return pd.Series(smd)

smd_before = compute_smd(df_unmatched, covariates)
smd_after = compute_smd(matched_df, covariates)
```

**Visualize with a Love Plot:**

```python
plt.figure(figsize=(10, 6))
# Plot lines showing improvement
for i, cov in enumerate(covariates):
    plt.plot([abs(smd_before[cov]), abs(smd_after[cov])], 
             [i, i], 'gray')

plt.scatter(abs(smd_before), range(len(covariates)), 
           label='Before', s=100, color='red')
plt.scatter(abs(smd_after), range(len(covariates)), 
           label='After', s=100, color='green')
plt.axvline(0.1, linestyle='--', label='Threshold')
plt.yticks(range(len(covariates)), covariates)
plt.xlabel('|Standardized Mean Difference|')
plt.title('Love Plot: Balance Before and After Matching')
plt.legend()
```

**Interpretation:**
- Points moving left (toward 0) = improved balance
- All points left of 0.1 threshold = excellent balance

---

### Step 7: Estimate Average Treatment Effect on Treated (ATT)

```python
def estimate_att(matched_df):
    # Calculate outcome difference within each pair
    treated = matched_df[matched_df.role=='treated'].sort_values('pair_id')
    control = matched_df[matched_df.role=='control'].sort_values('pair_id')
    
    differences = treated['purchase'].values - control['purchase'].values
    att = differences.mean()
    
    # Bootstrap 95% CI
    boot_atts = []
    for _ in range(1000):
        sample = np.random.choice(differences, size=len(differences), 
                                 replace=True)
        boot_atts.append(sample.mean())
    
    ci = (np.percentile(boot_atts, 2.5), np.percentile(boot_atts, 97.5))
    
    return att, ci

att, ci = estimate_att(matched_df)
print(f"ATT: {att:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
```

**Example Output:**
```
ATT: 0.078 (95% CI: [0.064, 0.092])
```

**Interpretation:** The discount increased purchase probability by 7.8 percentage points (95% CI: 6.4% to 9.2%). Since our true effect is 8%, our estimate is accurate!

---

### Step 8: Robustness Checks

#### A. Inverse Probability Treatment Weighting (IPTW)

Alternative estimator that reweights observations:

```python
def estimate_iptw(df):
    # Weight = Treatment/PS + (1-Treatment)/(1-PS)
    df['weight'] = (df.treatment / df.propensity_score + 
                    (1 - df.treatment) / (1 - df.propensity_score))
    
    # Trim extreme weights
    df['weight'] = np.clip(df['weight'], 0.1, 10)
    
    # Weighted means
    treated_mean = np.average(df[df.treatment==1]['purchase'], 
                             weights=df[df.treatment==1]['weight'])
    control_mean = np.average(df[df.treatment==0]['purchase'], 
                             weights=df[df.treatment==0]['weight'])
    
    return treated_mean - control_mean

ate_iptw = estimate_iptw(df_model)
```

**Compare estimates:**
- PSM ATT: 0.078
- IPTW ATE: 0.081

Similar estimates strengthen confidence in causal effect.

#### B. Vary Caliper

Try different caliper values:

```python
calipers = [0.1, 0.05, 0.02, 0.01]
for c in calipers:
    pairs = nearest_neighbor_match(df_model, caliper=c)
    att, _ = estimate_att(matched_df)
    print(f"Caliper {c}: ATT = {att:.3f}, N = {len(pairs)} pairs")
```

**Stable estimates** across calipers → robust finding.

#### C. Alternative Propensity Models

Try random forest or boosting for propensity estimation. Compare balance and ATT.

---

## Interpreting Results {#interpretation}

### What does ATT mean?

**ATT = Average Treatment Effect on the Treated**

"Among customers who were offered the discount, the discount increased their purchase probability by X percentage points."

**Important distinctions:**
- **ATT:** Effect for those who received treatment
- **ATE:** Average effect if everyone received treatment
- **ATU:** Average effect for those who didn't receive treatment

PSM with 1:1 matching typically estimates ATT.

### Statistical vs. Practical Significance

- **Statistical:** Is the effect distinguishable from zero? (Check CI)
- **Practical:** Is the effect large enough to matter?

Example: ATT = 0.078 (7.8 percentage points)
- If baseline purchase rate is 27%, this is a 29% relative increase
- If discount costs 10% margin and increases purchases 7.8%, calculate ROI

### Causality Claims

PSM enables causal claims **only under key assumptions:**

1. **Ignorability:** All confounders are observed and included in propensity model
2. **Positivity:** There's overlap in propensity scores
3. **SUTVA:** No interference (one customer's treatment doesn't affect another)

**Limitations:** If there are unobserved confounders (hidden variables affecting both treatment and outcome), PSM estimates remain biased.

---

## Common Pitfalls {#pitfalls}

### 1. Omitted Variable Bias

**Problem:** Forgot to include an important confounder

**Example:** Didn't include "previous purchase history" which affects both discount assignment and future purchases

**Solution:** 
- Carefully consider all variables affecting treatment and outcome
- Use domain knowledge and causal diagrams (DAGs)
- Consider sensitivity analyses

### 2. Poor Overlap

**Problem:** Treated and control propensity distributions don't overlap

**Symptoms:**
- Very few matched pairs
- Extreme propensity scores (near 0 or 1)

**Solutions:**
- Trim propensity scores (e.g., 0.1 < PS < 0.9)
- Re-specify propensity model
- Accept reduced generalizability

### 3. Propensity Model Overfitting

**Problem:** Used complex model (deep tree, many features) that perfectly predicts treatment

**Symptoms:**
- Many propensity scores = 0 or 1
- IPTW weights explode

**Solutions:**
- Use regularized models
- Cross-validate propensity model
- Prefer simpler models (logistic regression)

### 4. Post-Treatment Variables

**Problem:** Included variables affected by treatment in propensity model

**Example:** If discount affects engagement, don't condition on post-discount engagement

**Solution:** Only include pre-treatment covariates

### 5. Ignoring Balance Diagnostics

**Problem:** Proceeded with analysis despite poor balance (large SMDs)

**Solution:** 
- Always check balance before estimating effects
- If balance is poor, try different caliper, matching ratio, or propensity model
- Consider other methods (coarsened exact matching, full matching)

---

## Exercises {#exercises}

### Exercise 1: Basic (Naive vs. Adjusted Comparison)

Compute the naive difference in purchase rates between treated and control (no matching). Compare it to your PSM ATT estimate. What does the difference represent?

**Expected findings:** Naive estimate should be biased upward due to selection on high-propensity customers.

---

### Exercise 2: Intermediate (Sensitivity to Caliper)

Change the caliper to 0.02 and re-run the matching. How many pairs remain? How does ATT change? Interpret.

**Guiding questions:**
- Does stricter matching improve balance?
- Is the trade-off in sample size worth it?

---

### Exercise 3: Advanced (Alternative Propensity Model)

Use `RandomForestClassifier` to estimate propensity scores:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                  random_state=42)
rf_model.fit(X, y)
propensity_rf = rf_model.predict_proba(X)[:, 1]
```

Compare balance and ATT to the logistic model. Which performs better? Why?

**Challenge:** Check if random forest creates extreme propensity scores. If yes, what would you do?

---

### Exercise 4: Real-World Application

Apply PSM to a dataset of your choice (e.g., job training programs, medical treatments, marketing campaigns). Document:

1. Research question and treatment
2. Potential confounders identified
3. Propensity model specification
4. Balance diagnostics
5. Estimated effect with interpretation
6. Limitations and assumptions

---

## Additional Resources {#resources}

### Foundational Papers

1. **Rosenbaum & Rubin (1983)**: "The Central Role of the Propensity Score in Observational Studies for Causal Effects"
   - Proves that conditioning on propensity score balances covariates

2. **Austin (2011)**: "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding"
   - Comprehensive tutorial with practical guidance

3. **Stuart (2010)**: "Matching Methods for Causal Inference: A Review"
   - Reviews various matching approaches

### Books

- **Pearl, J. (2009)**. *Causality: Models, Reasoning, and Inference*
  - Formal framework for causal inference

- **Morgan & Winship (2015)**. *Counterfactuals and Causal Inference*
  - Accessible introduction to causal methods

- **Hernán & Robins (2020)**. *Causal Inference: What If*
  - Modern textbook, freely available online

### Python Libraries

- **CausalML**: Uber's library for causal inference
- **EconML**: Microsoft's library with multiple estimators
- **DoWhy**: Microsoft library for causal reasoning with graphs

### Online Courses

- Coursera: "A Crash Course in Causality" (Penn)
- EdX: "Causal Diagrams" (Harvard)

---

## Conclusion

You've learned how to:
✅ Estimate causal effects from observational data using PSM
✅ Check assumptions and diagnose problems
✅ Interpret results and assess robustness
✅ Avoid common pitfalls

**Key Takeaway:** PSM is a powerful tool, but remember the assumptions. Always check balance, conduct robustness checks, and be transparent about limitations.

**Next Steps:**
1. Practice with different datasets
2. Learn complementary methods (IV, diff-in-diff, RD)
3. Study causal diagrams (DAGs) for better variable selection
4. Consider doubly-robust methods (combine propensity and outcome models)

---

*Happy causal inferring! 🎯*





Video Link : [https://drive.google.com/file/d/1_M_Wyt5YXDC1mkHDoCKstN8Sb6c8ig3N/view?usp=drive_link](https://youtu.be/HRcwRn5gjwg)
