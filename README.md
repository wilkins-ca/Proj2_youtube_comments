# Analyzing Youtube Comment and Video Data in US Videos

## This project analyzes US-based Youtube comment and video data to explore commenter sentiment, emoji usage, video engagement patterns, and trending topics. It leverages Python for preprocessing and cleaning the data, sentiment and emoji analysis, topic modeling, and custom engagement scoring, with Power BI used for final visualizations and storytelling. This project began as a guided exercise from a Udemy course on data analytics. I followed the core structure but significantly extended it by adding original components, such as a combined approach to emoji and sentiment analysis, custom video engagement scoring, and unsupervised topic modeling using NMF.

--------

### Tools and Techniques
 --> The following Python libraries were used: pandas, TextBlob, emoji, collections, sklearn, matplotlib, seaborn, os, and string.
 --> The following techniques were applied: data wrangling, sentiment analysis, emoji frequency, custom scoring, TF-IDF vectorization, and NMF topic modeling
 --> Two word clouds from the wordcloud library, exported .csv files for use in Power BI visualizations, and finally, Power BI visualizations

### Project Steps
#### 1. Data collection and Preparation
 - parsed and filtered US-based Youtube comment data and video data
 - checked and delt with null values by dropping where appropriate
 - preprocessed text: cleaning, lowercasing, removing stopwords, and extracting emojis

#### 2. Sentiment and Emoji Analysis
 - used TextBlob for polarity scoring
 - created boolean has_emoji column
 - compared sentiment of emoji vs. non-emoji comments
 - counted most common emojis overall
