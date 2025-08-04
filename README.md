# Analyzing Youtube Comment and Video Data in US Videos

## This project analyzes US-based Youtube comment and video data to explore commenter sentiment, emoji usage, video engagement patterns, and trending topics. It leverages Python for preprocessing and cleaning the data, sentiment and emoji analysis, topic modeling, and custom engagement scoring, with Power BI used for final visualizations and storytelling. This project began as a guided exercise from a Udemy course on data analytics. I followed the core structure but significantly extended it by adding original components, such as a combined approach to emoji and sentiment analysis, custom video engagement scoring, and unsupervised topic modeling using NMF

--------

### Tools and Techniques

 --> The following Python libraries were used: pandas, TextBlob, emoji, collections, sklearn, matplotlib, seaborn, os, spaCy, tqdm and string
 --> The following techniques were applied: data wrangling, sentiment analysis, emoji frequency, custom scoring, TF-IDF vectorization, and NMF topic modeling
 --> Two word clouds from the wordcloud library, a heat map, a bar chart

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

#### 3. Topic Modeling

- used spaCy to lemmatize comment text, and TfidfVectorizer and NMF to find the top words for 10 components
- refined preprocessing method with custom stopword list and increasing efficiency of tokenization
- assigned topics to the top words in each of the 10 components

#### 4. Video Data Analysis

- preprocessed video data, including removing nulls and putting together list of unique video categories and their names
- visualized video likes versus video category and assigned a custom video engagement score
- generated dataframe of average engagement score by video category, fed into bar chart
- generated correlation matrix fed into heat map

### Key Takeaways

    1. the sentiment difference between comments using emojis and those not using emojis is roughly 0.055, not significant
    2. Average engagement score of videos in music category is less than expected at 0.06 where the highest average engagement score was 0.09 for Nonprofits and Activism
    3. Third topic modeled leaned strongly towards politics

### What I Learned

    1. Introduced the concepts of stopwords, lemmatization, and topic modeling
    2. Practiced transformation chains such as .pipe() and .assign()
    3. How to identify and preserve separate data granularities (comment-level vs. video-level)
