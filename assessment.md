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
5. **hotel_classRounded** : This indicates hotel class i.e. if its 5,4.6,4.5,4,3.5,3,2.5,2,1.5,1
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

Since,usually there are multiple recommendation algorithms to choose from, it is imperative to properly evaluate them for their performance.
Other than that, after building the recommender, we would definitely want to know how good recommender is?

Some of the approaches could be:
- Accuracy of their predictions : MAE, MSE,RMSE, Precision@K, Recall@K, ROC-AUC curve etc. 
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
    

<p style="color:blue"><i> Write your answer here </i> </p>

##### 1.3.3
Can you make use of LLM in your model building? If yes, how?
> We will be evaluating your creativity and LLM knowledge in this subtask

<p style="color:blue"><i> Write your answer here </i> </p>

##### 1.3.4
If you have the opportunity to acquire additional data and features, which specific data and features do you anticipate obtaining to enhance the ranking model? Furthermore, could you elaborate on how you intend to leverage these additions?

<p style="color:blue"><i> Write your answer here </i> </p>



#### 1.4 Deployments
As a data scientist, you might not need to deploy the models on your own as there are other colleagues like MLOps engineer who will help you with that. Nevertheless, you will still need to have a grasp of how the deployment works to facilitate communication with other team so that you do not build a model which is not possible to be deployed.
Sketch a simple diagram on how the model you build can be deployed to the cloud (AWS/GCP/Azure). 
> You will need to include data sources for your model in the diagram.
> You can design either a batch prediction system (store model results in a database) or just a service based system (for e.g. web services)
> We will evaluate based on the feasibility of your model and the thoughts behind your design. So you can write down your thoughts on why you choose the infrastructure (For cost saving/higher throughput/higher availability)

<p style="color:blue"><i> Write your answer here </i> </p>

<div style="page-break-after: always; visibility: hidden"> 
\pagebreak 
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

<p style="color:blue"><i> Write your answer here </i> </p>

#### 2.2 
##### 2.2.1 What is reinforcement learning? Can you give few examples of reinforcement learning?

<p style="color:blue"><i> Write your answer here </i> </p>

##### 2.2.2 In this case, what will be the ultimate metrics we can use to evaluate the performance of the pricing models?

<p style="color:blue"><i> Write your answer here </i> </p>


##### 2.2.3 If there are 4 algorithms/solutions developed by different data scientists, how can you design a fair experiment to compare the algorithms? Assume the product team has the capability to run the experiment you want

<p style="color:blue"><i> Write your answer here </i> </p>

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

- [ ] Bias is the error that occurs when a model consistently underestimates or overestimates the true value.
- [ ] Variance is the error that occurs when a model makes different predictions for the same input data.
- [ ] A model with high bias is underfitting, while a model with high variance is overfitting.
- [ ] All of the above.

#### 4.2 What is the purpose of data regularization?

- [ ] Data regularization is used to reduce the complexity of a model.
- [ ] Data regularization can help to prevent overfitting.
- [ ] Data regularization techniques include Ridge regression and Lasso regression.
- [ ] All of the above.

#### 4.3 What is the difference between parametric and non-parametric statistics?

- [ ] Parametric statistics assume that the data follows a specific distribution, while non-parametric statistics do not make any assumptions about the distribution of the data.
- [ ] Non-parametric statistics are more powerful than parametric statistics, but they are also more sensitive to violations of assumptions.
- [ ] Non-parametric statistics are more robust to outliers than parametric statistics.
- [ ] Non-parametric statistics requires large sample size to be valid

#### 4.4 Among the following identify the one in which dimensionality reduction reduces.

- [ ] Collinearity
- [ ] Information Gain
- [ ] Data Complexity
- [ ] Entropy


#### 4.5  What is the difference between bagging and boosting? Write down your opinion below.
<p style="color:blue"><i> Write your answer here </i> </p>

See more in *[appendix.md](./appendix.md)*
## The End

