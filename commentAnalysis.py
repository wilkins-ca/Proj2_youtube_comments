import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob as tb
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
import os
from warnings import filterwarnings as fwarn
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF as nmf
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as engStopwords
import spacy
from tqdm import tqdm


comments = pd.read_csv('UScomments_youtube.csv', on_bad_lines = 'skip')
# ^dataframe

# look at what we're dealing with
print(comments.columns)
# col titles are video_id, comment_text, likes, replies
print(comments.isnull().sum())
# 26 null values in the comment_text col. check how much that is compared to the size of the rest of the data
print(f"length of comment df = {len(comments)}")
# 691400 rows in df, compared to which 26 is insignificant. get rid of the null vals
comments.dropna(inplace = True)
print(f"post dropping dupes, length of comments df = {len(comments)}")

# sentiment analysis function
def sent_analyze(comment = str):
    return tb(comment).sentiment.polarity
    
# analyze sentiment for comment df
polarity = []
for comment in comments['comment_text']:
    try: # error checking
        polarity.append(sent_analyze(comment))
    except:
        polarity.append(0)

# add polarity col to comment df
comments['polarity'] = polarity
# add comment length col to comment df
comments['comment_length'] = [len(comm) for comm in comments['comment_text']]

# word cloud analysis
# filter to positive and negative sent. comments
pos_df = comments[comments['polarity'] >= 0.35]
neg_df = comments[comments['polarity'] <= -0.35]


stopwords = set(STOPWORDS)

# turn into one long string
total_pos = ' '.join(pos_df['comment_text'])
total_neg = ' '.join(neg_df['comment_text'])

# word cloud of pos comments
pos_wordcloud = WordCloud(stopwords = stopwords).generate(total_pos)
# word cloud of neg comments
neg_wordcloud = WordCloud(stopwords = stopwords).generate(total_neg)

# show word clouds
plt.imshow(pos_wordcloud)
plt.axis('off')
plt.show()
plt.imshow(neg_wordcloud)
plt.axis('off')
plt.show()

# extract emojis 
def emojiList(comment = str):
    return [char for char in comment if char in emoji.EMOJI_DATA]

all_emojis = []
for comment in comments['comment_text'].dropna(): # dropping missing vals
    all_emojis.append(emojiList(comment))

# add all_emojis col to comments df
comments['emojis_used'] = all_emojis

# count most common emojis and their frequency in list of tuples extracted from df
# flatten all_emojis, bc it's a list of lists
flat = [emoji for sublist in all_emojis for emoji in sublist]
mostCommon = Counter(flat).most_common(10)

# emoji only list
emojis = [mostCommon[i][0] for i in range(10)]

# frequency only list
freq = [mostCommon[i][1] for i in range(10)]

# plot emoji and freq
plt.bar(emojis, freq)
plt.title('Frequency of Top Emojis in Youtube Comments')
plt.show()

# create has_emoji col
comments['has_emoji'] = comments['emojis_used'].apply(lambda x: len(x) > 0)
# compare avg sentiment between comments w emojis and w/out
sentWEmoji = comments[comments['has_emoji'] == True]['polarity'].mean()
sentNoEmoji = comments[comments['has_emoji'] == False]['polarity'].mean()

print(f"avg sentiment of comments using emojis == {sentWEmoji}")
print(f"avg sentiment of comments without emojis == {sentNoEmoji}")
# with emojis = 0.19
# without emojis = 0.13


# export cleaned comment data to new csv
comments.to_csv('cleanedComments.csv')

# sentiment distribution for comments w and w/out emojis --> to be exported for a box plot in power bi
emojiSentDf = comments[['polarity', 'has_emoji']].copy()
emojiSentDf.to_csv('emojSentDist.csv')

# preprocess comments for use in topic modeling (Non-negative Matrix Factorization)
# preprocessing will include lowercasing, removing punctuation, removing stopwords, and tokenizing

def removePunc(comment):
    comment = re.sub(r"[^a-z\s]", " ", comment)
    comment = re.sub(r"\s+", " ", comment).strip()
    return comment


# create custom stopword list
extra = {'video','song','omg','amazing','nice','wow','page','funny','love','good','bad',
    'sad','crazy','cool','wtf','lol','lmao','eh','yay','subscribe','comment','views',
    'like','click','watch','content','link','share','music','album','lyrics','movie',
    'beat','clip','get','got','make','made','go','went','say', 'said','see', 'seen', 
    'think','know','feel','give','take','come','want','need','can','could','would',
    'i','you','he','she','we','they','me','him','her','us','them','this','that','these',
    'those','thing','stuff','someone','anyone','everyone','nobody', 'fuck','www','shit',
    'thank', 'does', 'hell','youtube', 'trending','does','did', 'fucking'} 
# ^tuned over the course of several runs to remove more stopwords

custom_stopwords = list(engStopwords.union(extra))

spacy_nlp = spacy.load("en_core_web_sm")
def lemmatize_text(text):
    doc = spacy_nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_ not in custom_stopwords])


# make lowercase
comments['lowercased'] = comments['comment_text'].str.lower()
# remove punctuation
comments['puncRmvd'] = comments['lowercased'].apply(removePunc)
# remove comments shorter than 5 words
comments = comments[comments['puncRmvd'].str.split().str.len() >= 5]
# lemmatize
tqdm.pandas()
comments['lemmatized'] = comments['puncRmvd'].fillna(' ').progress_apply(lemmatize_text)

# drop intermediate cols
comments.drop([['puncRmvd', 'lowercased']])

# vectorize
vectorizer = TfidfVectorizer(
    max_df = 0.95, 
    min_df = 5, 
    token_pattern = r'\b[a-zA-Z]{3,}\b', # only alphabetic words, 3+ characters
    max_features = 3000
)
tfidf_matrix = vectorizer.fit_transform(comments['lemmatized'])

nmf_model = nmf(n_components = 10, random_state = 42)
nmf_features = nmf_model.fit_transform(tfidf_matrix)
# get top words for each topic
features = vectorizer.get_feature_names_out()
nTopWords = 10
for topic_idx, topic in enumerate(nmf_model.components_):
    topWords = [features[i] for i in topic.argsort()[:-nTopWords - 1:-1]]
    print(f"topic {topic_idx}: {', '.join(topWords)}")

# the above works, but turns up "top" words for each of the 10 topics that 
# aren't very helpful, including explatives, abbreviations/exclamations like 'omg'
# and 'wow', and some emotionally neutral verbs. Therefore we will remove really short 
# comments ie less than 5 words (beginning line 153) and use a custom stopwords list 
# including common youtube lingo like subscribe, channel, and video

# comments['dominant_topic'] = np.argmax(nmf_features, axis = 1)
# print(comments['dominant_topic'].head())

# ============================================================================


# collect additional data from other csvs
files = os.listdir('csvs')
fwarn('ignore')

fulldf = pd.DataFrame()
path = 'csvs'
for file in files:
    current_df = pd.read_csv(path+'/'+file , encoding = 'iso-8859-1' , on_bad_lines = 'skip')
    fulldf = pd.concat([fulldf, current_df] , ignore_index = True)

# clean additional data
fulldf = fulldf.drop_duplicates()
print(f"length of full df post duplicate drop = {len(fulldf)}")
print(fulldf.isnull().sum())
print(f"Percent of lines with a null description = {fulldf.isnull().sum()['description'] / len(fulldf) * 100}")
# percent of whole which had null descript =~ 5%, proportionally insignificant, will drop
df0 = (fulldf.
       pipe(lambda d: d.dropna()))

print(type(df0))
print(f"len of clean df0 = {len(df0)}")
print(df0.columns)
# columns are video_id, trending_date, title, channel_title, category_id,
# publish_time, tags, views, likes, dislikes, comment_count, thumbnail_link,
# comments_disabled, ratings_disabled, video_error_or_removed, description

# export df0 to csv
# df0.to_csv('additionalData.csv')

# which category of youtube video has the most likes

# find unique category ids
ids = df0['category_id'].unique()
print(ids)

catdf = pd.read_json('jsons/US_category_id.json')
print(catdf.head())
# col names are kind, etag, items; each entry in items column is a list, with dict at index = 2 containing the info we want
# extract category names from items col
cat = {}
for item in catdf.values:
    cat[int(item[2]['id'])] = item[2]['snippet']['title']

# map category name to category id in df0
df0['category_name'] = df0['category_id'].map(cat)

# plot likes vs category name
plt.figure(figsize = (12,8))
sns.boxplot(x = 'category_name' , y = 'likes' , data = df0)
plt.xticks(rotation = 'vertical')


# define custom engagement score: comments weighted at 3, dislikes at 1.5, likes at 1
# additionally, add colunmn of score normalized by view count
def engagementScore(df):
    df = df.assign(
        engageScore = 3 * df['comment_count'] + 1.5 * df['dislikes'] + df['likes']
    )
    df = df.assign(
        normScore = (df['engageScore'] / df['views']).round(2)
    )
    return df

df1 = engagementScore(df0)

# generate df of avg engagement score (normalized by view count) by video category
avgEngByCat = (
    df1
    .groupby('category_name')['normScore']
    .mean()
    .round(2)
    .reset_index()
    .sort_values('normScore', ascending = False)
)

# export above avg df to csv for later use in power bi
# avgEngByCat.to_csv('AvgEngageByVidCat.csv')

# was going to merge df1 and comments on video_id to do a correlation matrix, but realized
# it's apples and oranges: df1 is data on the level of the video, and comment is just comment data, 
# e.g. the like col in df1 refers to likes on the video, whereas the like col in comments df refers 
# to likes on the comment. Therefore, let's do a correlation matrix focusing on video level data
cols = ['views', 'likes', 'dislikes', 'comment_count', 'comments_disabled', 'engageScore']
corr_df = df1[cols]
mat = corr_df.corr()
# export the matrix for later use in heatmap via power bi
mat.to_csv('vidDataCorr.csv')



#   TO DO NOTES
# run code, focus on lemmatization and topic modeling section