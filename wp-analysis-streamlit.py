# import streamlit as st
# import regex
# import emoji
# import re
# import datetime
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud, STOPWORDS
# import emoji
# import itertools
# from collections import Counter
# import warnings
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import opinion_lexicon
# import matplotlib
# import stopwords

# # Download NLTK resources (if not downloaded already)
# nltk.download('punkt')
# nltk.download('opinion_lexicon')
# nltk.download('vader_lexicon')

# # Suppress warnings
# warnings.filterwarnings('ignore')

# # Function to convert raw WhatsApp data to DataFrame
# def raw_to_df(file, key):
#     '''Converts raw .txt file into a Data Frame'''
    

#     split_formats = {
#         '12hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
#         '24hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
#         'custom' : ''
#     }
#     datetime_formats = {
#         '12hr' : '%d/%m/%Y, %I:%M %p - ',
#         '24hr' : '%d/%m/%Y, %H:%M - ',
#         'custom': ''
#     }

#     with open(file, 'r', encoding='utf-8') as raw_data:
#         raw_string = ' '.join(raw_data.read().split('\n'))
#         user_msg = re.split(split_formats[key], raw_string)[1:]
#         date_time = re.findall(split_formats[key], raw_string)

#         df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})

#     df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])

#     usernames = []
#     msgs = []
#     for i in df['user_msg']:
#         a = re.split('([\w\W]+?):\s', i)
#         if(a[1:]):
#             usernames.append(a[1])
#             msgs.append(a[2])
#         else:
#             usernames.append("group_notification")
#             msgs.append(a[0])

#     df['user'] = usernames
#     df['message'] = msgs

#     df.drop('user_msg', axis=1, inplace=True)

#     return df


# def detect_emojis(text):
#     emoji_list = []

#     # Define regex pattern to match emojis
#     emoji_pattern = re.compile("["
#                                u"\U0001F600-\U0001F64F"  # Emoticons
#                                u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
#                                u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
#                                u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
#                                "]+", flags=re.UNICODE)

#     # Iterate through each word in the text
#     for word in text.split():
#         # Find emojis in the word
#         emojis = re.findall(emoji_pattern, word)
#         # If emojis are found, append them to the emoji_list
#         if emojis!=None and emojis!=[''] and emojis!=[]:
#             emoji_list.extend(emojis)

#     # Return the count of emojis and the emoji_list
#     return len(emoji_list), emoji_list


# def identify_negative_words(text):
#     negative_words = set(opinion_lexicon.negative())
#     tokens = word_tokenize(text.lower())
#     negative_word_count = sum(1 for word in tokens if word in negative_words)
#     return negative_word_count, [word for word in tokens if word in negative_words]


# def identify_positive_words(text):
#     positive_words = set(opinion_lexicon.positive())
#     tokens = word_tokenize(text.lower())
#     positive_word_count = sum(1 for word in tokens if word in positive_words)
#     return positive_word_count, [word for word in tokens if word in positive_words]



# # Initialize sentiment analyzer
# sia = SentimentIntensityAnalyzer()

# # Streamlit app
# def main():
#     st.title('WhatsApp Data Analysis')
#     st.sidebar.title('Analysis Options')
#     df_real = raw_to_df('whatsapp-chat-data.txt', '12hr')
#     df = raw_to_df('whatsapp-chat-data.txt', '12hr')
#     df['day'] = df['date_time'].dt.strftime('%a')
#     df['month'] = df['date_time'].dt.strftime('%b')
#     df['year'] = df['date_time'].dt.year
#     df['date'] = df['date_time'].apply(lambda x: x.date())
#     df1 = df.copy()
#     df1['message_count'] = [1] * df1.shape[0]      # adding extra helper column --> message_count.
#     # df1.drop(columns='year', inplace=True)         # dropping unnecessary columns, using `inplace=True`, since this is copy of the DF and won't affect the original DataFrame.
#     # df1 = df1.groupby('date').sum().reset_index()
#     # sns.set_style("darkgrid")
#     # matplotlib.rcParams['font.size'] = 20
#     # matplotlib.rcParams['figure.figsize'] = (27, 6)

#     top10days = df1.sort_values(by="message_count", ascending=False).head(10)    # Sort values according to the number of messages per day.
#     top10days.reset_index(inplace=True)           # reset index in order.
#     top10days.drop(columns="index", inplace=True) # dropping original indices.
    

#     df2 = df.copy()
#     df2 = df2[df2.user != "group_notification"]
#     top10df = df2.groupby("user")["message"].count().sort_values(ascending=False)

#     # Final Data Frame
#     top10df = top10df.head(10).reset_index()
#     URLPATTERN = r'(https?://\S+)'
#     df['urlcount'] = df.message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
#     links=np.sum(df['urlcount'])

    
#     comment_words = ' '

#     # stopwords --> Words to be avoided while forming the WordCloud,
#     # removed group_notifications like 'joined', 'deleted';
#     # removed common words
#     stopwords = STOPWORDS.update(['group', 'link', 'invite', 'joined', 'message', 'deleted', 'yeah', 'hai', 'yes', 'okay', 'ok', 'will', 'use', 'using', 'one', 'know', 'guy', 'group', 'media', 'omitted'])


#     # iterate through the DataFrame.
#     for val in df2.message.values:

#     # typecaste each val to string.
#         val = str(val)

#         # split the value.
#         tokens = val.split()

#     # Converts each token into lowercase.
#     for i in range(len(tokens)):
#         tokens[i] = tokens[i].lower()

#     for words in tokens:
#         comment_words = comment_words + words + ' '
#     # Display raw data
#     if st.sidebar.checkbox('Show Raw Data'):
#         st.subheader('Raw WhatsApp Data')
#         st.write(df_real)

#     if st.sidebar.checkbox('Show Data with preprocessed date'):
#         st.subheader('Whatsapp Data with time and date when the messages sent')
#         st.write(df)
#     # Display basic statistics
#     if st.sidebar.checkbox('Show Basic Statistics'):
#         df["emoji"] = df["message"].apply(detect_emojis)
#         emojis = sum(df['emoji'].str.len())
#         st.subheader('Basic Statistics')
#         st.write('Total Messages:', df.shape[0])
#         st.write('Total Media Messages:', df[df['message'] == '<Media omitted>'].shape[0])
#         st.write('Total Emojis:', emojis)
#         st.write('Total Links:', links)
    

#     # For better readablity;
    
    
#     # Display messages per day plot
#     if st.sidebar.checkbox('Messages sent and not sent count from group'):
#         # st.subheader('Messages per Day Plot')
#         # plt.plot(df1.date, df1.message_count)
#         # plt.title('Messages sent per day over a time period')
#         # st.pyplot()
#         st.write("Total number of people who have sent at least one message on the group are " , len(df.user.unique()) - 1)  

#         st.write("Number of people who haven't sent even a single message on the group are ", 237 - len(df.user.unique()) - 1) 
    

#     # Display top 10 active users
#     if st.sidebar.checkbox('Show Top 10 Active Users'):
#         st.subheader('Top 10 Active Users')
#         st.write(top10df)

#     # Display sentiment analysis for a specific message
#     if st.sidebar.checkbox('Sentiment Analysis'):
#         st.subheader('Sentiment Analysis')
#         example_text_index = st.number_input('Enter message index:', min_value=0, max_value=len(df)-1, value=0)
#         text = df.loc[example_text_index, 'message']
#         sentiment_score = sia.polarity_scores(text)
#         negative_word_count, negative_words_list = identify_negative_words(text)
#         positive_word_count, positive_words_list = identify_positive_words(text)
#         sentiment_label = "Positive" if sentiment_score['compound'] > 0 else "Negative" if sentiment_score['compound'] < 0 else "Neutral"

#         st.write('Text:', text)
#         st.write('Sentiment Score:', sentiment_score)
#         st.write('Sentiment Label:', sentiment_label)
#         st.write('Negative Words Detected:', negative_words_list)
#         st.write('Positive Words Detected:', positive_words_list)

#     # Display word cloud
#     if st.sidebar.checkbox('Show Word Cloud'):
#         st.subheader('Word Cloud')
#         wordcloud = WordCloud(width=300, height=300, background_color='white', stopwords=stopwords, min_font_size=8).generate(comment_words)
#         st.image(wordcloud.to_image())

# if __name__ == '__main__':
#     main()


import streamlit as st
import regex
import emoji
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
import itertools
from collections import Counter
import warnings
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import opinion_lexicon
import matplotlib
import stopwords
from langdetect import detect 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
# Download NLTK resources (if not downloaded already)
nltk.download('punkt')
nltk.download('opinion_lexicon')
nltk.download('vader_lexicon')

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to convert raw WhatsApp data to DataFrame
def raw_to_df(file, key):
    '''Converts raw .txt file into a Data Frame'''
    split_formats = {
        '12hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
        '24hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        'custom' : ''
    }
    datetime_formats = {
        '12hr' : '%d/%m/%Y, %I:%M %p - ',
        '24hr' : '%d/%m/%Y, %H:%M - ',
        'custom': ''
    }

    with open(file, 'r', encoding='utf-8') as raw_data:
        raw_string = ' '.join(raw_data.read().split('\n'))
        user_msg = re.split(split_formats[key], raw_string)[1:]
        date_time = re.findall(split_formats[key], raw_string)

        df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})

    df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])

    usernames = []
    msgs = []
    for i in df['user_msg']:
        a = re.split('([\w\W]+?):\s', i)
        if(a[1:]):
            usernames.append(a[1])
            msgs.append(a[2])
        else:
            usernames.append("group_notification")
            msgs.append(a[0])

    df['user'] = usernames
    df['message'] = msgs

    df.drop('user_msg', axis=1, inplace=True)

    return df


def detect_emojis(text):
    emoji_list = []
    # Define regex pattern to match emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                               "]+", flags=re.UNICODE)

    # Iterate through each word in the text
    for word in text.split():
        # Find emojis in the word
        emojis = re.findall(emoji_pattern, word)
        # If emojis are found, append them to the emoji_list
        if emojis!=None and emojis!=[''] and emojis!=[]:
            emoji_list.extend(emojis)

    # Return the count of emojis and the emoji_list
    return len(emoji_list), emoji_list


def identify_negative_words(text):
    negative_words = set(opinion_lexicon.negative())
    tokens = word_tokenize(text.lower())
    negative_word_count = sum(1 for word in tokens if word in negative_words)
    return negative_word_count, [word for word in tokens if word in negative_words]


def identify_positive_words(text):
    positive_words = set(opinion_lexicon.positive())
    tokens = word_tokenize(text.lower())
    positive_word_count = sum(1 for word in tokens if word in positive_words)
    return positive_word_count, [word for word in tokens if word in positive_words]



# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit app
def main():
    st.title('WhatsApp Data Analysis')
    st.sidebar.title('Analysis Options')
    df_real = raw_to_df('whatsapp-chat-data.txt', '12hr')
    df = raw_to_df('whatsapp-chat-data.txt', '12hr')
    df['day'] = df['date_time'].dt.strftime('%a')
    df['month'] = df['date_time'].dt.strftime('%b')
    df['year'] = df['date_time'].dt.year
    df['date'] = df['date_time'].apply(lambda x: x.date())
    df1 = df.copy()
    df1['message_count'] = [1] * df1.shape[0]      # adding extra helper column --> message_count.
    
    df2 = df.copy()
    df2 = df2[df2.user != "group_notification"]
    top10df = df2.groupby("user")["message"].count().sort_values(ascending=False)

    # Final Data Frame
    top10df = top10df.head(10).reset_index()
    URLPATTERN = r'(https?://\S+)'
    df['urlcount'] = df.message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
    links=np.sum(df['urlcount'])

    comment_words = ' '
    stopwords = STOPWORDS.update(['group', 'link', 'invite', 'joined', 'message', 'deleted', 'yeah', 'hai', 'yes', 'okay', 'ok', 'will', 'use', 'using', 'one', 'know', 'guy', 'group', 'media', 'omitted'])


    # iterate through the DataFrame.
    for val in df2.message.values:

        # typecaste each val to string.
        val = str(val)

        # split the value.
        tokens = val.split()

        # Converts each token into lowercase.
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        for words in tokens:
            comment_words = comment_words + words + ' '
    
    # Display raw data
    if st.sidebar.checkbox('Show Raw Data'):
        st.subheader('Raw WhatsApp Data')
        st.write(df_real)

    if st.sidebar.checkbox('Show Data with preprocessed date'):
        st.subheader('Whatsapp Data with time and date when the messages sent')
        st.write(df)
    
    # Display basic statistics
    if st.sidebar.checkbox('Show Basic Statistics'):
        df["emoji"] = df["message"].apply(detect_emojis)
        emojis = sum(df['emoji'].str.len())
        st.subheader('Basic Statistics')
        st.write('Total Messages:', df.shape[0])
        st.write('Total Media Messages:', df[df['message'] == '<Media omitted>'].shape[0])
        st.write('Total Emojis:', emojis)
        st.write('Total Links:', links)
    

    # For better readablity;
    
    
    # Display messages per day plot
    if st.sidebar.checkbox('Messages sent and not sent count from group'):
        st.write("Total number of people who have sent at least one message on the group are " , len(df.user.unique()) - 1)  

        st.write("Number of people who haven't sent even a single message on the group are ", 237 - len(df.user.unique()) - 1) 
    

    # Display top 10 active users
    if st.sidebar.checkbox('Show Top 10 Active Users'):
        st.subheader('Top 10 Active Users')
        st.write(top10df)

    # Display sentiment analysis for a specific message
    if st.sidebar.checkbox('Sentiment Analysis'):
        st.subheader('Sentiment Analysis')
        example_text_index = st.number_input('Enter message index:', min_value=0, max_value=len(df)-1, value=0)
        text = df.loc[example_text_index, 'message']
        sentiment_score = sia.polarity_scores(text)
        negative_word_count, negative_words_list = identify_negative_words(text)
        positive_word_count, positive_words_list = identify_positive_words(text)
        sentiment_label = "Positive" if sentiment_score['compound'] > 0 else "Negative" if sentiment_score['compound'] < 0 else "Neutral"

        st.write('Text:', text)
        st.write('Sentiment Score:', sentiment_score)
        st.write('Sentiment Label:', sentiment_label)
        st.write('Negative Words Detected:', negative_words_list)
        st.write('Positive Words Detected:', positive_words_list)

    # Display word cloud
    if st.sidebar.checkbox('Show Word Cloud'):
        st.subheader('Word Cloud')
        wordcloud = WordCloud(width=300, height=300, background_color='white', stopwords=stopwords, min_font_size=8).generate(comment_words)
        st.image(wordcloud.to_image())

    # Perform topic modeling
    if st.sidebar.checkbox('Topic Modeling (LDA)'):
        st.subheader('Topic Modeling (LDA)')
        documents = df["message"]
        english_documents = []
        english_indices = []  # To store indices of English messages
        for idx, doc in enumerate(documents):
            try:
                if detect(doc) == 'en' and not any(char.isdigit() for char in doc):
                    english_documents.append(doc)
                    english_indices.append(idx)  # Store index of English message
            except:
                pass  # Ignore errors in language detection
        model_file=pickle.load(open('wp-analysis-model-file.pk1' , 'rb'))
        # Feature extraction
        vectorizer = CountVectorizer(max_df=0.85, min_df=3, stop_words='english')
        X = vectorizer.fit_transform(english_documents)

        # Model training (LDA)
        num_topics = 5  # You can adjust the number of topics as needed
        lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online')
        lda_model.fit(X)

        # Display the topics
        st.write("Topics found via LDA:")
        for idx, topic in enumerate(lda_model.components_):
            st.write(f"Topic {idx + 1}:")
            st.write([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])


            

if __name__ == '__main__':
    main()
