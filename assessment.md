# airasia Move Data Scientist Technical Assessment
v1.0.1

Welcome to the airasia Move Data Scientist Technical Assessment! This evaluation is designed to gauge your proficiency in key areas of data science and assess your problem-solving skills. As a data scientist, the tasks ahead will challenge you to apply your knowledge of **statistics, programming, machine learning, software engineering, data manipulation, software architecture, business acumen and data analysis.**

This assessment is structured to reflect real-world scenarios, providing you with an opportunity to showcase your analytical thinking and coding abilities. Approach each task thoughtfully and feel free to document your thought process as you work through the problems. **There is no sample answers for the given tasks** thus the evualuation is done by 3 random data scientist from the current team.

We do not mind you use any tools available in the Internet including Google, ChatGPT or expert advices as long as you can get the answers. **Only two out of the three questions (q1, q2, or q3) need to be completed.** If you encounter any problems or enquiries on the question set, you can always reach out to superappds@airasia.com on working hours or the recruiters who reach out to you.

Remember, the goal is not only to test your existing skills but also to provide insight into how you approach and solve data-related challenges. Best of luck and enjoy the journey of exploration and problem-solving!

<div style="page-break-after: always; visibility: hidden"> 
\pagebreak 
</div>

## Section A 
Candidates are expected to select either 2 out of the 3 tasks under this section.
### Task 1: Hotel
As selling hotels is part of airasia Move challenge, data scientists in will delve into a classic problem on determining the optimal ranking for a list of hotels in an area.

The final objective is to develop a robust algorithm that takes into account multiple factors influencing hotel ranking. From customer reviews and ratings to amenities and location, your task is to craft a solution that accurately reflects the quality and appeal of each hotel. **It can be personalized ranking, generalized ranking or a hybrid approach.**

> This challenge is designed to assess your ability on:
> 1. Business acumen on hospitality and travel industry
> 2. Ranking/recommendation algorithm
> 3. Creativity on feature engineering
> 4. Software engineering/solution architecture knowledge to probe feasibility of the solution provided by yourself
> 5. Understanding of data engineering concept.
> 6. Data wrangling/processing

#### 1.1 Data Extraction
Before building the ranking model, the first thing to do is to extract the data from the operational database or data warehouse. 
##### 1.1.1. What are the primary distinctions between an operational database and a data warehouse/lakehouse? Is it possible to perform data analysis on an operational database?
<p style="color:blue"><i> Answer: Operational database a.k.a transactional databases handle large amount of real-time transactional processing of records. They are specifically designed for OLTP (Online Transactional Processing). An example of such datbase can be Call detail records being stored in near-real-time in a telecom operator's database or retailer storing its inventory information.

On the other hand, Datawarehouses/lakehouses are specifically designed to perform large number of analytical operations. In other words, they are designed for **OLAP (Online Analytical Processing)** workloads. Example includes, Airline or Railway Booking Systems often use data warehouses to store and analyze booking data. Data warehouses are also used to store and process large amounts of relevant healthcare data.

Technically, it is possible to perform data analysis directly on an operational database. However, since operational databases are specifcially designed to perform efficient data insertion, updating, and retrieval to support the live operations of a business, it is not a recommended practice.

The reason being, long-running query or any such analytical operation on operational database can drastically slow down the application (for e.g. website front-end or CRM application) resulting in poor customer experience. Moreoever, the data in an operational database is constantly changing as new transactions are processed. This can lead to inconsistencies in the data analysis results.

Instead, often, 3 step process of Extract, Transform and Load (ETL) is performed to load the operational data into a datawarehouse. 

</i> </p>



##### 1.1.2. What features will you consider when developing an algorithm to rank hotel inventories?"
<p style="color:blue"><i> Since, we are missing explicit indicators like actual ratings given by users or even reviews data, We would have to rely on implicit indicators to build a recommendation and ranking system. Some of the features can be:
  
1. **NumBookings** : This denotes the total count of bookings received by a hotel over the full span from Mar to Sept.
2. **TotalRevenue** : This denotes the total revenue collected in USD.
3. **AvgMonthlyBooking** : The number in this feature indicates the Monthly average number of bookings a hotel has received. For instance, hotel with id '-9213121850607123932' has received total of 5 bookings in last 7 months, so its score is (5/7). While hotel with id '-9206097099752226690' has received 10 bookings in last 7 months, so its score is (10/7)
4. **AvgWeeklyBooking** : The number in this feature indicates the Weekly average number of bookings a hotel has received. This is similar in concept to AvgMonthlyBooking feature.
5. **hotel_class** : This indicates hotel class i.e. if its 5,4.6,4.5,4,3.5,3,2.5,2,1.5,1. (Missing classes were imputed using prediction algorithm). Please see notebok q1.3.1.ipynb
5. **Latestbookings_count** : This is the count of the number of bookings that a hotel has received in last 2 months. Recent bookings might indicate current popularity or relevance.

Apart from these, other features can be : amenities, star rating,clickThroughRate, views,reviews, sentiments and others
</i> </p>

##### 1.1.3. How do you conduct Exploratory Data Analysis (EDA) to validate whether the features support your hypothesis?
<p style="color:blue"><i> I conducted comprehensive EDA by finding a pattern in the number of bookings per hotel. Then explored temporal and seasonal booking trends using time-series chart. Also, analyzed from monetary point of view by dividing given data into multiple dollar value bins/categories and mapping total number of bookings agaist it, Finally also created scatter plot to confirm that the 'length of stay' in the hotel and the 'dollar amount' of booking are positively correlated. Please refer to the notebook q1.3.1.ipynb for code and graphs.  </i> </p>


#### 1.2 Data wrangling
Please find `hotel_sample_data_raw_bookings.csv` and `hotel_sample_data_raw_inventories.csv`  in the `data` folder. Use either SQL or python to aggregate the data to **hotel level** with the metrics below:
> Refer to appendix.md for the data dictionary and description

1. MTD (month-to-date) total bookings (as of 2023-09-18)
2. Average daily booking
3. Average weekly booking
4. Average monthly booking
5. Second last booking date, null if total bookings <= 1
6. Recency (as of 2023-10-01)
7. Average transaction value

You just need to answer in either `q1.2.ipynb` or `q1.2sql`.

#### 1.3 Data modelling
##### 1.3.1
Please make use of the data you have processed in the `1.2` to build a sample ranking model.
> We understand that the best model is not built in a day with limited features. Thus this task is only evaluating on your coding abilities and will not evaluate much on modelling abilities thus you do not need to spend too much time to optimize the model to get best performance

Save your working steps, codes and notes in a `output/q1.3.ipynb`.


##### 1.3.2
What can you do to measure & optimize the model performance? Provide your thoughts.
> We will be evaluating on your experience/knowledge on ranking/recommendation projects in this subtask.

<p style="color:blue"><i> Since,usually there are multiple recommendation algorithms to choose from, it is imperative to properly evaluate them for their performance.
Other than that, after building the recommender, we would definitely want to know how good recommender is?

Some of the approaches could be:
- Accuracy of their predictions : Computing MAE(Mean Absolute Error), MSE (Mean Squared Error),RMSE (Root Mean Squared Error), Precision@K, Recall@K, ROC-AUC curve etc.
  
- Usefulness of recommendations:
  - Correctness
  - Non-Obviousness
  - Diversity
- Computational performance.

Above measures are more 'model-focused' and consider each recommendation locally. 
But, the real-world involves considering user-preferences and their behavior.
So, we can also perform experiments:
- Evaluation with users:
  - User Surveys,Polls, Log analysis - I want to know what the user eventually selected and how they relate to recommendations i produced?
    We might ask users if they find recommendation useful or how often they rely on them. However, creating robust surveys is hard!
  - A/B testing (Controlled Lab experiments) :  We can give 100 people in each of 2 groups exact same experience except for some variable and we will measure if one of them prefers something or uses something differently than others. 
    
</i> </p>

##### 1.3.3
Can you make use of LLM in your model building? If yes, how?
> We will be evaluating your creativity and LLM knowledge in this subtask

<p style="color:blue"><i> LLMs are suitable for dealing with textual data. In this context, LLM can certainly be useful in enriching the user and item profiles. For instance, many times, user's leave a short review/feedback on hotels after their trip. Such information can be used by a language model to understand the sentiment of the review or to extract certain topics or entities from the review. 

More specifcially, sentiment analysis can be implemented to extract users’ attitude towards hotels as part of the input for the recommendation system. 

I have previously wrote a paper using this technique to recommend STEAM games to end users : [https://arxiv.org/ftp/arxiv/papers/2312/2312.11503.pdf](https://arxiv.org/ftp/arxiv/papers/2312/2312.11503.pdf)

**Conversational Recommendations:**

On top of this, LLM can be used to create a chat-like interface as part of which, it user query as input for instance, **"Can you recommend top-5 hotels to me?"**. Once input is received, the LLM which was supposedly fine-tuned using instruction-based prompt training to extract keywords from the input can then pass on the summary the recommender system backend. This could be in the form of API. Recommender system can use the information provided by the LLM, along with other relevant data (user history, item features, etc.), to generate personalized recommendations.
The recommendations are returned to the LLM, which can then craft a natural language response to present the recommendations to the user.

LLM can also be used to be a QA agent to answer questions on recommendations - We need to fine-tune the LLM on question-answer pairs based on sample recommendations generated.

LLM can also be used to summarize recommendations into a paragraph - We need to fine-tune the LLM on summary dataset generated out of recommendations generated by recommendation system.

Also, We can use generative AI to automatically generate or enhance item descriptions. This is particularly useful when the item database has limited or incomplete information. The generated descriptions can then be used as input features for the recommender system.

We can leverage generative AI to create additional content related to recommended items, such as blog posts, reviews, or short summaries. This content can be presented alongside recommendations to provide users with more context and information.

We can also use generative AI for continuous training and optimization by analyzing and generating insights from user feedback. The model can process user reviews, comments, and feedback to understand sentiment and extract valuable information for refining recommendations.

Previously I have used GEN-AI to create QA bot: https://github.com/netgvarun2012/DocumentAnalyzer

and also, finetuning GenAI to generate tips and recommendations based on emotional state of a person: 

https://github.com/netgvarun2012/VirtualTherapist

</i> </p>

##### 1.3.4
If you have the opportunity to acquire additional data and features, which specific data and features do you anticipate obtaining to enhance the ranking model? Furthermore, could you elaborate on how you intend to leverage these additions?


<p style="color:blue"><i> 
The current Recommendation system is indeed retrained by the amount of data and features available to us.
Like previously mentioned, we can be greatly benifited if more features relevant to this usecase can be collected. Such as:

1. **User Features**: In order to create a robust user profile, user information can be immensely useful. This could include demographic information (like age, gender, or location), behavioral information (like browsing history or click patterns), or any other user attributes that might influence their preferences.

2. **Hotel Features**: More detailed information about the hotels could also be helpful. This could include amenities (like free WiFi, parking, or breakfast), location details (like proximity to tourist attractions or transportation hubs), room details (like room size or bed type), or ratings and reviews from users.

3. **User-Item interaction details**: Additional details about past interactions could be useful. For example, the price paid for past bookings, whether the user left a review or not, or whether the user has booked the same hotel multiple times.

4. **Ratings**: User ratings can be used as explicit feedback. For example, if a user consistently rates certain types of hotels highly, these hotels can be prioritized in the recommendations for that user.

5. **Review Text**: The text of a review can be analyzed to understand the sentiment of the review, or to extract key topics or features that the user comments on. This can be done using natural language processing (NLP) techniques. For example, if a user often mentions enjoying hotels with good views in their reviews, hotels with good views can be recommended to that user.

6. **Review Volume**: The number of reviews a hotel has received can be an indicator of its popularity, and can be used as a feature in the recommendation algorithm.

</i> </p>



#### 1.4 Deployments
As a data scientist, you might not need to deploy the models on your own as there are other colleagues like MLOps engineer who will help you with that. Nevertheless, you will still need to have a grasp of how the deployment works to facilitate communication with other team so that you do not build a model which is not possible to be deployed.
Sketch a simple diagram on how the model you build can be deployed to the cloud (AWS/GCP/Azure). 

![IMG_8578](https://github.com/netgvarun2012/AirlineUseCase/assets/93938450/d5c9d03d-a334-400b-8b9b-4affce2615f9)


> You will need to include data sources for your model in the diagram.
> You can design either a batch prediction system (store model results in a database) or just a service based system (for e.g. web services)
> We will evaluate based on the feasibility of your model and the thoughts behind your design. So you can write down your thoughts on why you choose the infrastructure (For cost saving/higher throughput/higher availability)
<p style="color:blue"><i> 
In the diagram above, I have shown an 'Online Prediction' system as part of which, predictions are generated and returned as soon as the requests for these predictions arrive.
When doing online prediction, requests are sent to prediction service via RESTful APIs.

In online prediction, both batch features and streaming features can be used. Features computed from historical data (such as data stored in DataWarehouse) are batch features. Whereas, Features computed from streaming data - data in real-time transports- are streaming features.

An example of batch feature used for online prediction, especially session-based recommendations, is item embeddings. Item embeddings are pre-computed in batch and stored in DW and are fetched whenever they are needed for online prediction.

Since hotel availability and prices can change rapidly, and users’ preferences might also change based on various factors like location, time, budget, etc. A web-service-based system can provide real-time recommendations based on the most current data.

Moreover, A web-service-based system can take into account the user’s current context and recent interactions to provide more personalized recommendations.

The choice of deployment design should be guided by the specific requirements of the system, the user experience we want to provide, and the resources available for maintaining the system.

So, we can also look at **Batch Prediction system (asynchronous)** which is used by Netflix to generate movie recommendations for all of its users every 4 hours. Recommendations generated by the recommender system are stored in a **DataWarehouse** and precomputed recommendations are fetched and shown to the user when they logon to Netflix. 

To summarize:
- If **High throughput** is of importance, then Batch predictions system should be preferred.
- If **Low Latency** is required, then Online prediciton system should be preferred.

</i> </p>

<div style="page-break-after: always; visibility: hidden"> 
 
</div>


### Task 2: Pricing
Dynamic pricing is one of the challenge we have in airasia Ride. We have multiple challenges when designing the algorithm.

1. We must strike a delicate balance between driver and passenger satisfaction. High fares may please drivers but drive away passengers, while low fares may attract passengers but disgruntle drivers. 
2. Pricing should be tailored to each city's unique characteristics. A one-size-fits-all approach would be ineffective as living cost in every cities are different.

> This challenge is designed to assess your ability on:
> 1. Business acumen on revenue and pricing
> 2. Dynamic Pricing
> 3. Geographical data science
> 4. Solution Architecture
> 5. Machine Learning
> 6. Experiment Design

#### 2.1.Solution Design
Share us your thoughts on how you will design the pricing system.
Write down/sketch your thoughts on how you will design the algorithm and how it can be deployed.

<p style="color:blue"><i>
Dynamic pricing is a broad strategy that involves adjusting prices in real-time based on various factors, such as demand, supply, competitor pricing, and other market conditions.

Let's take an example use-case of Uber!

_During periods of excessive demand or scarce supply i.e. when there are far more riders than drivers Uber increases its normal fares with a multiplier whose value depends upon the scarcity of available drivers._

Concepts from good old Microeconomics is used to calculate the market price for riders and drivers alike.

In more technical terms, the Goal of surge pricing is to find the ‘equilibrium price’ at which driver supply matches rider demand and rider’s wait time is minimized. This is because Uber can not afford to have a scenario when demand > supply at crunch times in major cities!

**Modelling**:
- Suppose there are ‘n’ locations.

**On demand side:**
- Demand at location ‘i’ :
```
    Di ⇒ ai - bi pi
```
Here, the variable 'p' typically represents the price of the product or service, and 'D' represents the quantity demanded.
In this specific model, the negative sign in front of ‘b’ indicates an inverse relationship between price and quantity demanded. This is consistent with the law of demand, which states that, all else being equal, as the price of a good or service increases, the quantity demanded decreases, and vice versa.

Below formulation is a demonstration for 2 locations:

**On supply side:**
- When building a supply model, one has to take into account the relationship between 2 locations:

```
S1 = c1 + dP1 - Q12 P2 (When price at say Clementi (Singapore) goes up, supply at NUS goes down, so negative relationship).
S2 = c2 + dP2 - Q21 P1
```

The positive sign for coefficient 'd' indicates a direct or positive relationship between the price of a product and the quantity that suppliers are willing to produce and sell. This is consistent with the general expectation in economics: as the price of a good or service increases, suppliers are often willing to supply more of that good or service to the market, all else being equal.

Since the primary goal of dynamic pricing is to optimize revenue by setting prices that reflect changing market dynamics and consumer behavior, We need to define an objective function.

The objective function is formulated to maximize the revenue, which is represented by the sum of the products of prices and the minimum of demand and supply for two locations:

              maximize (P1 . min(D1,S1) + P2 . min(D2,S2) )
              
**Decision Variable**:
The decision variables are the prices for the two locations:
        ```  P1, P2 >=0 ```

**Constraints**:
Constraints involve the relationships between demand, supply, and pricing for each locations.

For Location 1:
           ``` S1 = c1 + d1P1 - Q12P2 ,  D1 = a1 - b1P1```
              
For Location 2:
         ```   S2 = c2 + d2P2 - Q21P1 ,  D2 = a2 - b2P2```

For solving such optimization problem, **Gurobi** is a popular commercial  solver known for its efficiency and performance. 
Other options include **IBM CPLEX, SciPy Optimization, Pyomo**.

In terms of machine learning, we can integrate this dynamic pricing optimization with a trained MachineLearning Model which can enhance the model's ability to adapt to complex, non-linear relationships, and changing market conditions.

- For that , We need to gather historical data on prices, demand, and other relevant factors.
- Identify relevant features that can influence pricing decisions. These may include demand drivers, seasonality, competitor prices, and any other factors that impact your pricing strategy.
- Model selection : Regression model, time-series model, reinforcement model cam worl out well.
- Train the model.
- Validation and deployment.

We can develop separate pricing models or parameters for each city. This involves training machine learning models independently for each city or adjusting coefficients in your optimization model based on city-specific characteristics.

Assuming the city-specific regression models are 'reg_model_cityA' and 'reg_model_cityB'
```
predicted_prices_cityA = reg_model_cityA.predict(new_data_cityA)
predicted_prices_cityB = reg_model_cityB.predict(new_data_cityB)
```

### Integrate predicted prices into optimization model for each city
```
model.setObjective(predicted_prices_cityA[0] * min(D1, S1) + predicted_prices_cityA[1] * min(D2, S2) +
                   predicted_prices_cityB[0] * min(D1, S1) + predicted_prices_cityB[1] * min(D2, S2), sense=GRB.MAXIMIZE)
```

In terms of deployment, We can Choose a platform for deploying the pricing model. This could be a cloud platform like AWS, Azure, or Google Cloud, or an on-premises server. Cloud platforms often provide convenient services for deploying and managing machine learning models. We can expose our model through an API (Application Programming Interface) so that other systems or applications can make requests to our pricing model. This enables real-time communication between your pricing model and the systems that need pricing decisions.

We can also think of implementing A/B testing to evaluate the performance of the dynamic pricing system. Test different pricing strategies on a subset of users to gather insights into the impact of changes before deploying them broadly.

For comprehensive deployment strategy, please take a look at the deployment sketch above section 1.4

</i> </p>

#### 2.2 
##### 2.2.1 What is reinforcement learning? Can you give few examples of reinforcement learning?

<p style="color:blue"><i> 

Reinforcement learning involves an Agent which has a complete view of the situation within which it is called the STATE. 
Agent also has a partial view of the state which is given to it as input from the environment at each timestep. 
This is known as an observation. Agent interacts with the environment and takes an action. It gets a reward or a penalty as a result of taking that action.

The goal of reinforcement learning then is to learn a policy function that maps states to action in such a way that cumulative reward is maximized.

More formally:

- Sequential interaction of an agent with its environment. ​​
- At each time step _t_, the agent observes RL environment of internal state _St_ and retrieves observation _Ot_. ​​
- Executes the action at resulting from its RL policy _π(at |ht)_ where ht is the RL agent history and receives a reward rt as consequence of its action.​​
- Policies are designed as a function that maps states to actions and maximizes an optimality criterion.​
- Directly depends on the immediate rewards _rt_ observed over a certain time horizon.​

I have previously worked on **Re-inforcement learning** for stock trading use case. More specifically:

"_To create a stock timing model, spontaneously find the best trading opportunity to complete the trading and strive for the lowest overall trading cost of the stock._"

I have written a medium article on it previously:
https://medium.com/@sharmavarun.cs/deep-reinforcement-learning-for-stock-trading-90c6f63d3439 

Github Repo:
https://github.com/netgvarun2012/StockTradingDRL

Apart from Stock Trading example above, Reinforcement learning has been used as RLHF models (Reinforcment Learning with Human Feedback) to correct LLMs that behave badly for e.g. exhibiting toxic language, aggressive responses, dangerous information. 
RL has been famously used in game-playing scenarios, such as the success of AlphaGo, also to train self-driving cars to navigate complex environments, in robotics to teach robots how to perform tasks such as grasping objects, manipulating tools,
</i> </p>

##### 2.2.2 In this case, what will be the ultimate metrics we can use to evaluate the performance of the pricing models?

<p style="color:blue"><i> When it comes ot metrics, usually, model centric classification/regression metrics comes to mind. However, it is important to also consider business objective build metrics around it.

  1. **Profit Maximization**: Since one of the primary goals of the pricing model is to maximize profit, one can measure the total profit before and after implementing the model1.
  
  2. **Sales Volume**: Dynamic pricing models can help increase sales volume by adjusting prices in response to changes in market demand1. Therefore, tracking changes in sales volume can be a useful metric.
  
  3. **Customer Satisfaction**: Dynamic pricing can improve customer satisfaction by providing consumers with prices that are more in line with their perceived value of a service1. One could measure this through customer surveys or by tracking repeat business.
  
  4. **Market Share**: If one of the objectives of your dynamic pricing model is to outbid competitors and get a larger share of the market2, then market share could be a relevant metric.
  
  5. **Accuracy Metrics**: Depending on the machine learning techniques used, one might also consider standard predictive accuracy metrics such as Percent Correct Classification (PCC) or confusion matrix4.
  
  6, **A/B Test Results**: If applicable, one can measure the impact of different pricing strategies using A/B testing.

</i> </p>


##### 2.2.3 If there are 4 algorithms/solutions developed by different data scientists, how can you design a fair experiment to compare the algorithms? Assume the product team has the capability to run the experiment you want

<p style="color:blue"><i> 
  In such a case, we would have to devise a fair experiment to compare 4 different algorithms developed by data scientists and it will require careful consideration to ensure unbiased results and valuable insights.  
  
- First step would be to clearly dedine the goal of the experiment. These can be done by asking questions like : are we trying to identify the algorithm with the highest accuracy, best recall, or lowest processing time?
  
- Then, We need to choose appropriate metrics to quantify the performance of each algorithm based on the objective. Ensure the metrics are well-defined, objective, and relevant to the business problem.
  
- All algorithms should be tested on the same dataset. This ensures that any differences in performance are due to the algorithms themselves and not variations in the data.
  
- Implement blinding. This means that individuals involved in the experiment, such as data scientists, analysts, or evaluators, are unaware of which algorithm corresponds to which solution. This minimizes bias.
  
- Ensure that your results are statistically significant. Use appropriate statistical tests to assess the significance of observed differences between algorithms.
  
- Document the experiment design, methodology, and any decisions made throughout the process.
  
- Consider factors like:
  - Deploying the chosen algorithm in production. Factors like computational cost, infrastructure requirements, and maintainability should be taken into account.
  - Interpretablity - allowing one to understand why they make certain predictions.
    
- Communicate the experiment design and progress with the product team and relevant stakeholders
</i> </p>

<div style="page-break-after: always; visibility: hidden"> 
\pagebreak 
</div>

### Task 3: Demand Prediction
If we can forecast the demand of e-hailing orders, it will be very helpful for the operation team to arrange the drivers in the correct location to improve completion rate and utilize the fleet.

> This challenge is designed to assess your ability on:
> 1. Supervised learning & unsupervised learning
> 2. Forecasting algorithm
> 3. Geographical data science
> 4. Critical thinking
> 5. Machine Learning
> 6. Business communication
> 7. Project management

#### 3.1 Modelling
##### 3.1.1
Use `data/ride_sample_data_raw_bookings.csv` to build a demand prediction model including EDA, feature engineering, and data wrangling. 
> Refer to `appendix.md` for the data dictionary and description.

Jot down your working steps and code in  `output/q3.1.1.ipynb`. We will be only evaluating on the steps/approaches but not on the performance of your model.


##### 3.1.2 
How would you address outliers in the data arising from special events such as concerts and festival seasons?
<p style="color:blue"><i> Write your answer here </i> </p>


##### 3.1.3 
Assuming you are required to present your solution to a **business stakeholder** for this use case. Prepare a presentation slide deck consisting of 2-3 pages to effectively pitch the solution to them.
> Note: The model is not deployed thus we need the business stakeholders to buy our idea so that this project can be prioritized in the product roadmap. You will also need to provide how it can be used by the product or operation team.

Prepare your slide in `output/q3.1.3.pptx`


## Section B (Compulsory)
### 4 Data Science Foundation

#### 4.1 What is the difference between bias and variance?

- [x] Bias is the error that occurs when a model consistently underestimates or overestimates the true value.
- [ ] Variance is the error that occurs when a model makes different predictions for the same input data.
- [x] A model with high bias is underfitting, while a model with high variance is overfitting.
- [ ] All of the above.

#### 4.2 What is the purpose of data regularization?

- [ ] Data regularization is used to reduce the complexity of a model.
- [ ] Data regularization can help to prevent overfitting.
- [ ] Data regularization techniques include Ridge regression and Lasso regression.
- [x] All of the above.

#### 4.3 What is the difference between parametric and non-parametric statistics?

- [x] Parametric statistics assume that the data follows a specific distribution, while non-parametric statistics do not make any assumptions about the distribution of the data.
- [ ] Non-parametric statistics are more powerful than parametric statistics, but they are also more sensitive to violations of assumptions.
- [x] Non-parametric statistics are more robust to outliers than parametric statistics.
- [x] Non-parametric statistics requires large sample size to be valid

#### 4.4 Among the following identify the one in which dimensionality reduction reduces.

- [x] Collinearity
- [ ] Information Gain
- [x] Data Complexity
- [ ] Entropy


#### 4.5  What is the difference between bagging and boosting? Write down your opinion below.
<p style="color:blue"><i> 

Techniques of bagging and boosting have originated from Decision Trees. Decision trees are fast to train, and can have low bias but high variance

### BAGGING (e.g. Random Forest) 
To train, fix some number T, then:
• Bootstrap T times, then train T decision trees, one on each bootstrap

- We can further reduce variance by reducing the correlation between the trees.

- Random Forest does this correlation by :
  - For each tree (trained on each bootstrapped sample),every time we choose a split, we only allow splits along some random subset of m features.
 
### BOOSTING (e.g. AdaBoost - Adaptive Boosting)
We are boosting our weak models by adapting to the samples we got wrong. 
Gradient Boosting generalizes AdaBoost by allowing different loss functions (original ADABOOST has exponential loss function).

The key contribution of boosting is that it provides a way of training and combining weak learners to turn them into a single strong learner.

Some key differences are listed below:

| Bagging                                            | Boosting                                     |
| ---------------------------------------------------| -------------------------------------------- |
| 1. This involves training many Indeoendently grown | 1. This involves training many sequentially  |
| trees in parallel.                                 | grown trees.                                 |
| 2. This involves training strong learners(low bias)| 2. This involves trainsing weak learners     |
| e.g. deep decision trees.                          | (high bias) e.g. shallow decision trees.     |
| 3. This reduces Variance.                          | 3. This reduces bias.                        |
| 4. This handles over-fitting                       | 4. This handles Underfitting                 |
| 5. More trees doesn’t hurt                         | 5. Too many trees leads to over-fitting!     |
</i> </p>

See more in *[appendix.md](./appendix.md)*
## The End

