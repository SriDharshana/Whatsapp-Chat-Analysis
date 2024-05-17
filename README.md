# Whatsapp-Chat-Analysis
A comprehensive analysis tool for WhatsApp chat data leveraging Python libraries and natural language processing techniques is presented. The application is developed using Streamlit, a user-friendly web framework, to provide an interactive and visually appealing interface for exploring various aspects of chat conversations. 

# Model and Libraries used
The Whatsapp chat analysis pipeline begins with preprocessing raw WhatsApp data, including message extraction, user identification, and timestamp parsing. Employed regular expressions for efficient text parsing and NLTK for advanced text processing tasks such as sentiment analysis, emoji detection, and word frequency analysis. Additionally, used utilize word clouds to visualize word frequencies and identify prominent themes within the chat corpus

Multinnomial Naive bayes - Accuracy : 85% , Linear SVC - Accuracy : 80%, NLTK, langdetect, LatentDirichletAllocation, opinion_lexicon, Sentiment Intensity Analyzer

# Datasets

Whatsapp chat from any one of the groups can be used as a dataset in Whatchat Analysis Project
 
**Original Dataset**




**Preprocessed Dataset**

![image](https://user-images.githubusercontent.com/86719672/210181808-22cbd2b1-5233-47d4-8e69-bd2b8fa6a6a8.png)

**Confusion Matrix**

![image](https://user-images.githubusercontent.com/86719672/210181924-782fc15a-a104-4fef-9a52-fc4f6f0d2326.png)


# Prediction

Sentimental analysis detects the positive words and negative words used in a particular chat

Text: guys, please do me a favor and vote in this poll. or if you can, please share it. thank you.   

**Output**

Sentiment Score: {'neg': 0.0, 'neu': 0.556, 'pos': 0.444, 'compound': 0.875}

Sentiment Label: Positive

Negative Words Detected: []

Positive Words Detected: ['favor', 'thank']

