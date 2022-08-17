# Introduction

This is a project aiming to offer a simple framework for defining and estimating churn in a multi-product systems where the churn behavior is not obvious . 

**How to identify churn?**

Unlike some other contexts such as credit cards where churn be identified easily, in many other systems such as retail the churn behavior is not obvious. Let's say you are a retail store company that carries products from many vendors. One of your vendors is interested to know the churn probability of its products. A customer could have been a regular customer and not buy that product for awhile. ***There's no definite way to tell if that customer has churned or not.*** Therefore, churn has to be inferred from the data.

In order to infer churn for a product (or brand), we need to consider the entire product category, as churn usually implies switching to other competitor brands. By considering all the possible purchase behaviors with respect to a brand, we can infer the churn probability.

Here, we offer a framework to segment any market based on the target brand and brand category. The framework yields 7 segments that can be used in churn probability estimation. For any brand, all the possible purchasing behaviors can be segmented as follows:

| Brand / Category      | Brand's Category      | Brand's Category & Competing Categories   | Competing Categories Only         |
|:----------------|:----------:|:-----------:|:-----------:|
|only Target Brand        |  segment1 |  segment4  |  segment7  |
|Target Brand & Other brands|  segment2 |  segment5  |  segment7  |
|Other Brands Only        |  segment3 |  segment6  |  segment7  |

<br>

For example, consider the E-cigarette brand JUUL. It's in the E-cigarette category and the competing category is Cigarettes. Therefore, the following segments can be inferred:

| Brand / Category      | E-cigarettes Only    | E-cigarettes + Cigarettes   | Cigarettes Only         |
|:----------------|:----------:|:-----------:|:-----------:|
|only JUUL        |  segment1 |  segment4  |  segment7  |
|JUUL + Other brands|  segment2 |  segment5  |  segment7  |
|Other brands        |  segment3 |  segment6  |  segment7  |

<br>

This framework can be similarly applied to any other brand. Therefore, for any target brand, we could imagine seven segments that describe all the possible purchasing behaviors.

**Note 1:** One of the possibilities is that a customer abandons the entire product category (e.g. when a customer quits smoking). We consider such cases as in "segment7" as it doesn't change the estimations.

**Note 2:** The above categories can be further split based on ***usage*** (heavy, moderate, light).

**How to handle sparsity?**

It is often the case that the customer purchase journeys are sparse. For example, a customer may not purchase any product from a particular brand for a long time. In such cases, we can define time-based thresholds to define the customer as a churner or not. For example, a customer is considered a churner if he/she has not purchased any product from a brand for more than 3 months.

**Note:** You could experiment with different time-based thresholds to fine-tune the one that works best for your system. 

**Note:** The chosen threshold can also be verified by looking at historical data. For a specific threshold, we can see how many of customers labeled as churners actually churned.

**Note:** Moreover, it might also be helpful to distinguish between died/alive states with respect to a specific brand purchases vs. general purchases.

| Brand(X)\General| Alive           | Died              |
|:----------------|:----------------|:------------------|
|Alive            |  Strongly Alive |  Brand (X) Alive  |
|Died             |  Brand (X) Died |  Strongly Died    |

**How do we determine being Alive?**

- Made at least 1 purchase in the past “x” months/time intervals
- $\text{P(Alive)} = f(\text{tenure}, \text{ \# of trans in tenure})$ $\rightarrow$ this definition would fit better as measure of level of being active.


<hr>

# Transition Probability

An important aspect of defining a churn is identifying signals of changing behavior. In that sense, examining customers switching patterns between different segments can be helpful in identifying those signals. Given the segments, we can quantify the movements across different segments and calculate transition probability between them at each time interval (e.g. month).

In fact, we can directly estimate transition probability as a function of some time-dependent/independent covariates. It can be modeled as log odds of transitioning from one segment to another, given the current segment:

$$\pi_{ss^\prime} = \log\frac{p(s_t = s | s_{t-1} = s^\prime)}{p(s_t = s^\prime | s_{t-1} = s^\prime)} = \alpha_i + f(\mathbf{X}_t) + g(\mathbf{Z}) + \varepsilon$$

<hr>

## How to model churn?

Transition probability matrix model can’t be used directly for predicting churn since:
1. Depending on the Markov order (assuming Markov transition), we would only be able to predict $t+1$. But most of churn cases are concerned about probability of being alive in the next $N$ time intervals.
2. There is no alive/died segment (segment8 is just not observed, doesn’t mean “died”).

There are two different ways we could model churn:
1. Model the probability of churn in the next $N$ time intervals (e.g. in the next 6 months)
2. Model the time to churn (***survival analysis***: time until the occurrence of an event)

Following approach (1), the log-likelihood function is,

$$LL = \sum_{it} \log [p(\text{churn}_i = 1 |s_t=s) \times \pi_i(s_t=s|s_{t-1}=s^\prime) \times \pi_i(s_0=s)]$$

In approach (2), we should define a hazard function (probability of some event in next t time interval/ probability of staying alive until then):

$$h(x) = \frac{P(x)}{P(X > x)}$$

$P(X > x)$ is called **survival function**, which is related to distribution function: $P(X < x) = 1 - P(X > x) = 1 - \text{survival function}$. For including features in hazard function, we could use Cox proportional hazards regression model.

**Potential features**
- RFM-based features
- customer vectors
- metrics based on amount/number of transaction
- specific product purchases (smoking cessation, buying other brands, buying cig for someone who always bought e-cig, etc.)

<hr>

# Implementation
Below, we'll provide some more details on how the source code works. Please note that almost all the codes are written in PySpark, so you'll need Spark to run them.

## Data preparation
In order to capture switching behavior between segments, we'd preferably like to assign segment membership dynamically, so that we could examine segment switching behavior . We use customer transactions data as our main data source. To get segments dynamically, we need to split data into separate chunks based on the time interval (determined by user) and add segment labels for each customer.

All the data preparation methods are collected under the `DataLoader` class. To get the final data you just need to run the `transform_data` method. It gets the raw customer transaction data as input as outputs the a spark `DataFrame` which contains segments as well as the features that we'll use in training our model.

More specifically, we do the following steps to transform raw transaction data into its final format:
1. **Chunking data:** Based on the time interval (i.e. `window`: time interval length in months, and `step`: step size for rolling window) determined by user, we split data into chunks. This is done by `gen_data_chunks` method which is generator that iterates over data based on `window` and `step`.
2. **Adding segment labels:** After splitting data into chunks, we apply the "segmenting logic" to each chunk to add segment label for each customer. This is done by `get_segment_data`.
3. **Equalize observations for each customer:** One of downsides of using transaction data is that we don't (necessarily) observe customer for the entire period. We only observe them when they make a purchase. However, in understanding customer switching behavior (and eventually churn behavior), we also need to account for those unobserved times. `balance_segments` method will do that. It'll add `null` segment rows for unobserved occasions.
4. **Impute segments for unobserved times:** For transition probability estimation, in case, adding the `null` segment rows to data makes it sparse, we impute them. We _**assume**_ that if a customer doesn't show up, they still belong to their last-seen segment. Therefore, we forward-fill the `null` cells with their last-seen segment. We wrote a customized class, `Impute`, which contains different imputation methods and use that class here. The `impute_nulls` in `DataLoader` class uses the imputation class and fill in the empty cells in data.
5. **Adding segment at $t-1$:** We model transition probabilities as log odds model of conditional probabilities. We predict segment at time **t** given that we know it at **t-1**. `add_segment_prev` method adds a column with the value of segment at previous time period. It'll drop the first observation for each customer, since the previous segment label is unknown.
6. **Adding features:** In addition to previous segment, we'd also like to add some more features. We, first, use `get_dummies` method to convert the categorical columns into dummy variables. The `aggregate_trans_data` method calculates transaction-related features for each customer. Eventually, `get_aggregate_trans_data` uses our `gen_data_chunks` generators to calculate and add features for each time interval.
7. **Joining segment and transaction features:** To get the final data for model training, we join the segment labels data frame with the features. This is done by `join_trans_segment`.

## Segmenting Logic
Although, the project is mainly defined for Juul company, but the source codes are agnostic to the brand (Juul in this case) at hand so that the same code can be used any other brand (X) with almost no change. The only part that needs to be changed is what we call **"segmenting logic"**. We have a separate module to collect all the segmenting logics. `DataLoader` will automatically picks the logic determined by user, with no change.

## Training & Evaluation

Please refer to the notebooks folder for an example of training and evaluating a transition probability model.

***

>**Further Study**
>
>- Survival-based models seems to be the best options as we're dealing with censored data (customer's status is NOT known throughout all observation points).
>- The above segments can be thought of as hidden states, a hidden Markov model where there is a transition probability matrix for transitioning between different states (segments).
>- Basket analysis of segments, do we see in difference across different segments? -> are customers in different segments inherently different? Their basket combo should be statistically different.
>  - We can use customer data (non-trans)
>- One interesting aspect of the problem is to characterize switching behavior. For example, smoking is habitual, so switching shouldn't be that common (verify with data).
>- Customer journey can modeled using graph theory, where nodes are 2D vertices (segment, time). This representation may help in better visualization of customer journey (customer journey = customer transitioning between different segments through time).
>- Sometimes strong churn signals can be inferred from the data. For example, buying from **smoking cessation** category might be a good signal of churning from smoking category.
>- Churn rate for each customer should eventually be moderated by the customer's lifetime value. In another words, it’s not efficient to focus retention efforts only based on churn rates. We wouldn’t worry much when a customer churns if she has very small CLV. On the other hand, in terms of retention, we’d very like to retain (i.e. keep churn rate low) for customers with very high CLV. CLV here should be brand-specific, meaning that if we’re estimating churn rate for JUUL, we’d estimate the CLV with respect to JUUL as well.

