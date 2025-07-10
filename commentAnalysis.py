import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob as tb
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
import plotly.graph_objs as go


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

# word cloud analysis
# filter to positive and negative sent. comments
pos_df = comments[comments['polarity'] == 1]
neg_df = comments[comments['polarity'] == -1]


stopwords = set(STOPWORDS)

# turn into string
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

# count most common emojis and their frequency in list of tuples extracted from df
# flatten all_emojis, bc it's a list of lists
flat = [emoji for sublist in all_emojis for emoji in sublist]
mostCommon = Counter(flat).most_common(10)

# emoji only list
emojis = [mostCommon[i][0] for i in range(10)]

# frequency only list
freq = [mostCommon[i][1] for i in range(10)]

# plot emoji and freq

# PUT THE BELOW IN PLT, NOT PLOTLY.GO
trace = go.Bar(x = emojis, y = freq)
layout = go.Layout(title = 'Emoji Frequency')
fig = go.Figure(data = [trace], layout = layout)
fig.show()