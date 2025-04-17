# Readme
Below are the approach,challenges and potential improvements

APPROACH:

Data loading:
The data was loaded into a data frame . 

Data cleaning and analysis:
Then the data was analyzed for nulls using info function.
It was found that the features column contained the information for some of rows but majority of them were nulls . The pros_list column also had details for the capabilities . So this column was used incase the feature column was null. 5 rows had null rating.Those were replaced by 0

Feature selection:
The important columns were picked for further analysis - vendor,software_name,categories,features,pros_list 
The features column was processed and converted to pick only the relevant information .The pros_list was also processed similarly .Finally both were combined into 1 column 


Similarity Scoring:
In this step ,for the selected capability the goal was to filter the vendors who have atleast one of the expected features. TF-IDF and Cosine similarity were used to achieve this .
Additionally threshold based filtering was used to ensure filtering does not remove relevant data 


Vendor Ranking:
The final stage involves ranking the vendors based on a combination of Weighted Feature Similarity, overall vendor rating (if available), and Final Score Calculation. 
Weighted Feature Similarity: This ensures more weight is given to vendors with more relevant features.
Vendor Rating Incorporation: If vendor ratings are available, they are incorporated with a weight assigned based on their importance. 
Final Score Calculation: A final score is calculated for each vendor by combining the weighted similarity score and the rating score (if used).
Rank Generation: The vendors are ranked based on their final scores, with higher scores resulting in higher ranks. The ranked list is presented to the user as the final output.
If no capabilities are entered the final result is based on vendor rating alone


CHALLENGES:

Extracting the features involved lot of processing 
Lot of null values 

POTENTIAL IMPROVEMENTS:
Advanced NLP techniques
In depth data exploration
Data visualization
Use additional fields like reviews,social media profiles,feedback,rating split,pricing etc 
User interface 
Metrics 
Include more test cases 
