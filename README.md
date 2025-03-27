### **Rreview Analyzer**

The aim of the project is to develop a simple AI-driven solution for analysing reviews, which includes a combination of NLP techniques with a pre-trained large language model (LLM) and computer vision for image analysis. The results are displayed in the `final` folder.

1️⃣ **Part #1 - Data Collection and Preprocessing**
    The data was scraped using APIFY through the  add-on "Restaurant Review Aggregator" add-on(`tri_angle/restaurant-review-aggregator`)
    in batches and collected in CSV files ans are located in the `data` folder.

2️⃣ **Part #2 - Text Analysis using LLM**
    The following two LLM models have been used in this section:
    1. `tabularisai/robust-sentiment-analysis` - for labelling semtiments
    2. `distilbert-base-uncased` - to train the model for semtiment analysis

3️⃣ **Part #3 - Image Analysis (Vision AI)**
    For the classification of images, I have used two methodologies:
    1. CLIP (Contrastive Language-Image Pre-Training)
    2. Batch Classification with ResNet

4️⃣ **Part #4 - Combining Insights**
    Did my own interpetaion with building a "chat-sql" with streamlit.
    This is a chat where you are able to ask "sql style" questions regarding the dataset.
    ***usage:***
    To use the application, execute the `app.py` file using the Streamlit CLI. Make sure you have Streamlit installed before running the application. Run the following command in your terminal:
   ```
    stamlit run app.py
    ```