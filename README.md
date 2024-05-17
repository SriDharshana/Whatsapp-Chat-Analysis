# Whatsapp-Chat-Analysis
A comprehensive analysis tool for WhatsApp chat data leveraging Python libraries and natural language processing techniques is presented. The application is developed using Streamlit, a user-friendly web framework, to provide an interactive and visually appealing interface for exploring various aspects of chat conversations. 

The Whatsapp chat analysis pipeline begins with preprocessing raw WhatsApp data, including message extraction, user identification, and timestamp parsing. Employed regular expressions for efficient text parsing and NLTK for advanced text processing tasks such as sentiment analysis, emoji detection, and word frequency analysis. Additionally, used utilize word clouds to visualize word frequencies and identify prominent themes within the chat corpus

NLTK, langdetect, LatentDirichletAllocation, opinion_lexicon, Sentiment Intensity Analyzer, stopwords, emoji count vectorizer were used

**Topic Modelling**

   - Latent Dirichlet Allocation (LDA) or similar topic modeling techniques are applied to identify latent topics within the chat data.
   - The text data is vectorized using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) to prepare it for topic modeling.
   - The number of topics is determined based on domain knowledge or through techniques like grid search.
     
# Datasets

Whatsapp chat from any one of the groups can be used as a dataset in Whatchat Analysis Project

# Dataset preprocessing

   - The raw chat data, typically stored in a text file, is loaded into memory using Python's file handling capabilities.
   - Regular expressions are utilized to parse timestamps and extract user messages from the raw text data.
   - Media content and URLs are handled separately, with media messages being counted and URLs extracted for further analysis.
   - Data is organized into a structured format, typically a pandas DataFrame, for further processing.
   - Textual features such as word frequency, message length, and presence of emojis are extracted from the user messages.
   - Emojis are detected using regular expressions or specialized libraries like emoji.
   - Additional features such as the presence of URLs and media messages are also extracted.


 
**Original Dataset**

![image](https://github.com/SriDharshana/Whatsapp-Chat-Analysis/assets/86719672/33706976-9df8-4868-aba6-e78443402d29)



**Preprocessed Dataset**

                date_time                        user  \
0     2020-01-26 16:19:00          group_notification   
1     2020-01-24 20:25:00          group_notification   
2     2020-01-26 16:19:00          group_notification   
3     2020-01-26 16:20:00          group_notification   
4     2020-01-26 16:20:00          group_notification   
...                   ...                         ...   
13650 2020-10-02 02:05:00   Darshan Rander (TSEC, IT)   
13651 2020-10-02 02:05:00   Darshan Rander (TSEC, IT)   
13652 2020-10-02 02:11:00     Tanay Kamath (TSEC, CS)   
13653 2020-10-02 02:28:00   Darshan Rander (TSEC, IT)   
13654 2020-10-02 10:13:00  Dheeraj Lalwani (TSEC, CS)  

                                                 message              emoji  \
0      Messages and calls are end-to-end encrypted. N...            (0, [])   
1      Tanay Kamath (TSEC, CS) created group "CODERS👨...  (3, [👨, 💻👩, 💻🖥💻])   
2             You joined using this group's invite link             (0, [])   
3      +91 99871 38558 joined using this group's invi...            (0, [])   
4      +91 91680 38866 joined using this group's invi...            (0, [])   
...                                                  ...                ...   
13650                                    MCQs mark kiya             (0, [])   
13651                                    Sign-in kiya😂😅           (1, [😂😅])   
13652                                  Incognito se na?             (0, [])   
13653                                               Yup             (0, [])   
13654  guys, please do me a favor and vote in this po...            (0, [])   

       urlcount  day month  year        date  
0             0  Sun   Jan  2020  2020-01-26  
1             0  Fri   Jan  2020  2020-01-24  
2             0  Sun   Jan  2020  2020-01-26  
3             0  Sun   Jan  2020  2020-01-26  
4             0  Sun   Jan  2020  2020-01-26  
...         ...  ...   ...   ...         ...  
13650         0  Fri   Oct  2020  2020-10-02  
13651         0  Fri   Oct  2020  2020-10-02  
13652         0  Fri   Oct  2020  2020-10-02  
13653         0  Fri   Oct  2020  2020-10-02  
13654         1  Fri   Oct  2020  2020-10-02  

[13655 rows x 9 columns]



# Prediction

Sentimental analysis detects the positive words and negative words used in a particular chat

Text: guys, please do me a favor and vote in this poll. or if you can, please share it. thank you.   

**Output**

Sentiment Score: {'neg': 0.0, 'neu': 0.556, 'pos': 0.444, 'compound': 0.875}

Sentiment Label: Positive

Negative Words Detected: []

Positive Words Detected: ['favor', 'thank']

**Count Analysis**

Messages: 13117
Media: 0
Emojis: 27310
Links: 459

**WordCloud**

![image](https://github.com/SriDharshana/Whatsapp-Chat-Analysis/assets/86719672/e58f4585-22b0-4abc-bba3-f08e7f36bebc)

**Message count for each day**

![image](https://github.com/SriDharshana/Whatsapp-Chat-Analysis/assets/86719672/547ff15f-d45e-4dd8-876b-32c9b03077a5)





