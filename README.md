Flight Delay Prediction
============================================
Requirement for package are in requirements.txt

Data/ is where I stored all the dataset which are used in this project.

function/ is where I stored some function which re-use again.

Project.pdf is the PDF summary on the work flow and what i can improve.

flight_delay_model.py contain the model which i used to generate the result at the end and other model which i tested. The end model at the end is pickle out

others .ipynb is test code which i sued to test the dataset and some idea i would like to try (maybe i didnt use it at the end due to different reason). 



============================================
You will be given a learning data set with flight delay claims from 2013/01 to 2016/07. The goal is to predict claim amount for the future flights in hidden data set. Please use any machine learning techniques to make your prediction as accurate as possible. We judge model quality Q using mean absolute error Q1 and mean squared (Brier error) error Q2 between the predicted claim and actual claim.


gift: Bonus Point: we welcome you to add extra parameters/data to the learning & testing data test for more precise prediction. This could become one of the evaluation criteria for us (since it demonstrates how good you are in terms of crawling other useful data).

Business Objectives
A higher amount of predicted claim, with the cap of $800 as an arbitrary value, should be assigned to the high-risk flights. This is to adequately compensate the risk we need to take and naturally screen out the high risks flights.

A lower amount of predicted claim, as low as 0, should be assigned to the low-risk flights. This is to increase the conversion for a low-risk customer to buy and expand the risk pool.

The optimization should result in a very low absolute error |Expected(is_claim)–Actual(is_claim)| in an aggregated manner, which means we are not over/under-charging the customer (i.e. what we call precisely & dynamically price the risk for each customer).

Judging Criteria
The claim logic for Flight Delay Refund is:

If the delay_time field is greater than 3 hours OR equal to ‘Cancelled’, $800 will be claimed. Otherwise, claim amount will equal to $0.
We will calculate the result by taking average Absolute error Q1 and Brier error Q2 (should be your key optimization target) among all testing data.
Deliverables
Upload your source code to either GitHub or Bitbucket. Feel free to use any programming language and libraries.
We would download and run your code. Please make sure it is executable and there are instructions on how to setup the environment. In general, we use following guidelines to assess the submission.
Prepare a short ppt/pdf presentation with your thought about the problem, description of your approach and next steps, e.g. anything you did not implement or possible improvement areas. We would mainly judge the content of the presentation, please focus on that instead of its design or visual effects.
