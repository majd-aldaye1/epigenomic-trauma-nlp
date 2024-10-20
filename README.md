﻿# CAP-Epigenomics-Analysis_ma798_mmm443
## README

# Project: Epigenomic Analysis

This project analyzes how mental health disorders impact epigenetic markers, with additional breakdowns across ethnographic and socioeconomic disparities. The pipeline includes bulk data collection from PubMed, preprocessing of text data, topic modeling using Latent Dirichlet Allocation (LDA), and visualization using an interactive dashboard created in Dash.

---

### Demo Script
To run the demo:

1. **Clone the repository**:
   ```bash
   git clone git@github.coecis.cornell.edu:cs4701-24fa-projects/CAP_Epigenomics-Analysis_ma798_mmm443.git
   ```
2. **Go to the right directory**:
   ```
   cd .\CAP_Epigenomics-Analysis_ma798_mmm443\
   ```
4. **Install dependencies**:
   Ensure all required dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```
3. **Fetch data from PubMed**:
   - Run the PubMed fetching script:
     ```bash
     python scripts/pubmed_fetch.py
     ```
   - Verify the `data/pubmed_articles.csv` file is generated.
4. **Preprocess data**:
   - Run the preprocessing script:
     ```bash
     python scripts/preprocessing.py
     ```
   - Verify the `data/preprocessed_pubmed_articles.csv` file is generated.

5. **Run topic modeling**:
   - Run the topic modeling script:
     ```bash
     python scripts/topic_modeling.py
     ```
   - Verify that the topics are printed in the terminal and the interactive LDA visualization is saved as `models/lda_visualization.html`. Then feel free to open lda_visualization.html in a web browser to visualize the LDA topic modeling across the abstracts.

6. **Visualize using Dash**:
   - Run the Dash app for visualization:
     ```bash
     python scripts/main.py
     ```
   - Open your browser and navigate to `http://127.0.0.1:8050/` to view the interactive dashboard.

---

#### Team Members & Responsibilities:

**Member 1: Maya** (mmm443)
Responsibilities:  
- Data Fetching articles in bulk from PubMed related to epigenetics and mental health
- Developing the visualizations in main.py to create interactive interfaces for users to visualize the data across multiple dimensions
- Researching relevant terms for fetching data and preprocessing

**Demo + Steps to Run:**
1. **PubMed Data Fetching** (`pubmed_fetch.py`):
   - Maya will demonstrate how the script collects PubMed articles related to mental health, epigenetics, and socioeconomic factors, expands terms using PubMedBERT, and saves the results into a CSV file.
   
   **Steps to run:**
   - Navigate to the `scripts` directory:
     ```bash
     cd scripts
     ```
   - Run the PubMed fetching script:
     ```bash
     python pubmed_fetch.py
     ```
   - Verify that the `data/pubmed_articles.csv` file has been generated, containing relevant articles fetched from PubMed.

2. **Visualization in Dash** (`main.py`):
   - Maya will also show how to visualize the processed data using an interactive Dash dashboard that displays relationships between mental health, epigenetics, race, and socioeconomic status.

   **Steps to run:**
   - Ensure all data processing has been completed (e.g., running the preprocessing steps as explained below).
   - Run the Dash app:
     ```bash
     python main.py
     ```
   - Open your browser and navigate to `http://127.0.0.1:8050/` to see the interactive visualizations.
   
**TA Should Expect:**
- The TA will see how PubMed data is fetched, saved, and preprocessed into a csv file, and then how this data is visualized in the Dash dashboard across mental health and racial disparities.

---

**Member 2: Majd** (ma798)
Responsibilities:  
- NLP analysis for preprocessing abstracts, including fixing negation handling and keyword weights
- Visualizing model analysis through LDA topic modeling and fine-tuning model to biomedical terms
- Fixing numpy, scipy, pandas, and pyLDAvis dependencies and ensuring reproducibility across devices

**Demo:**
1. **Preprocessing** (`preprocessing.py`):
   - Majd will demonstrate how the `preprocessing.py` script cleans, tokenizes, lemmatizes, and categorizes the PubMed abstracts into meaningful categories (mental health terms, epigenetic terms, ethnographic terms, and socioeconomic terms).

   **Steps to run:**
   - Ensure `data/pubmed_articles.csv` exists from the previous step.
   - Run the preprocessing script:
     ```bash
     python preprocessing.py
     ```
   - Verify that the `data/preprocessed_pubmed_articles.csv` file has been created, which contains cleaned and categorized abstracts.

2. **Topic Modeling** (`topic_modeling.py`):
   - Majd will show how LDA topic modeling is performed on the preprocessed abstracts, generating topics and visualizing the topic distribution across documents.

   **Steps to run:**
   - Run the topic modeling script:
     ```bash
     python topic_modeling.py
     ```
   - Verify the output by viewing the topics printed in the terminal and inspecting the `models/lda_visualization.html` file, which can be opened in a browser to view the interactive topic visualization.

**TA Should Expect:**
- The TA will see the raw data get cleaned, categorized, and then passed through the LDA topic model. The TA can also view the interactive LDA visualization saved as an HTML file.

---

### General Notes

- **Data Setup**: If the dataset (`data/pubmed_articles.csv` or `data/preprocessed_pubmed_articles.csv`) is missing at any step, you can rerun the relevant fetching or preprocessing scripts as described above.
- **Running the Scripts**: Each script in the pipeline can be run independently for demonstration. Make sure the output files (CSV files, model artifacts, visualizations) are successfully generated at each stage before moving to the next demo step.
- **Simplified Demo Datasets**: For the sake of the demo, we use a smaller dataset to ensure fast execution of scripts and reduce the need for heavy computational resources during training.
- **Output Verification**: After running each script, check the respective CSV files and visualization outputs to verify that the process is working as expected.

---

### Additional Instructions

- **Interactive pyLDAvis**: 
   - Open `models/lda_visualization.html` in your browser to explore the LDA topics interactively.
- **Changing Parameters**: You can adjust the number of topics in the LDA model by modifying the `num_topics` argument in `topic_modeling.py`.

This demo showcases a full end-to-end analysis, from data collection and preprocessing to topic modeling and interactive visualizations.
