**DS interview technical case**

### Challenge 
Create a Machine Learning Model to Predict Customer

### ChurnObjective

The objective of this challenge is to create a machine learning model that can accurately predict customer churn. 

Churn refers to the act of a customer stopping their use of a product or service. This is an important problem for businesses as it can lead to a decrease in revenue and customer loyalty.

### Data

You will be provided with two datasets containing:
1. customer information along with various features such as customer demographics, account details; 
2. user activity history on the platform. 

Note that the dataset does not include a target variable indicating whether or not the customer churned. You can construct this target variable yourself depending on the datasets we provided.

| **Column**            | **Explanation**                                                              |
|-----------------------|------------------------------------------------------------------------------|
| ID                    | unique ID of a buyer                                                         |
| preferred_language    | preferred language of the user                                               |
| is_company            | if the user is a company                                                     |
| platform              | platform the user is registred on                                            |
| country               | user country                                                                 |
| gender                | gender of a buyer                                                            |
| mobile_device         | device type of a buyer                                                       |
| start_date_of_week    | start date of the counting weeknumber                                        |
| week                  | weeknumber                                                                   |
| year                  | year                                                                         |
| unique_lots_viewed    | amount of unique lots viewed by the user that week                           |
| amount_views          | total amount of view events by the user that week                            |
| bidded_on_amount_lots | total amount of unique lots the user bidded on that week                     |
| bids_places           | total amount of bids by the user that week                                   |
| total_bidded          | total amount of money the user bidded that week                              |
| money_spend           | total amount of money the user actually had to pay for lots he won that week |
| amount_lots_won       | total amount of lots the user won that week                                  |

### Task

Your task is to create a machine learning model that can accurately predict customer churn based on the provided data. You will need to preprocess the data, select relevant features, and train a model using a suitable machine learning algorithm.

### Requirements

- You will need to preprocess the data by handling missing values, scaling numerical features, and encoding categorical features.
- You will need to select relevant features that have a high correlation with the target variable.
- You will need to train a machine learning model using a suitable algorithm such as logistic regression, decision trees, or random forests.
- You will need to evaluate the model using metrics such as accuracy, precision, recall, and F1-score.
- You need to deploy your model as an API endpoint for model inference. It is optional to provide a web UI for demo purpose.

### Deliverable:

You will need to submit a report that includes the following:

- A summary of the dataset and the preprocessing steps taken.
- A list of the features selected and their importance.
- The machine learning algorithm used and the parameters tuned.
- The evaluation metrics used and their values.
- A discussion of the model's performance and any improvements that could be made.
- A functional, publicly accessible API endpoint to call the model and retrieve results.


http://93.43.208.11:8000/docs