**Project Title:** Steam Game Recommendation System

**Programme:** RDS Y2S3
**Tutorial Group:** 7
**Tutor:** Dr. Chuk Fong Ho
**Team members’ data:**
*   Lai Zhonpoa (2409205) - Collaborative Filtering / NN
*   Lee Jian Sheng (2408872) - Matrix Factorization
*   Alia Tasnim binti Baco (2411243) - Content-Based

---

**Section 1: Introduction**

**1.1 Problem Statement/Background**

The digital era is characterized by an unprecedented expansion of data, driven by the proliferation of internet-based technologies, social media, and mobile devices (Rypdal, 2018). Advanced computing enables the processing of these vast datasets, further fueling data generation and transforming industries globally. This data deluge significantly impacts the digital gaming industry. Platforms like Steam witness hundreds, even thousands, of new game releases monthly; Steam alone saw nearly 15,349 games published in the first few months of 2024, averaging over 350 per month (SteamSpy, 2024).

While this abundance offers players immense choice, it paradoxically leads to the "paradox of choice" (Dar & Gul, 2024), where an overwhelming volume of options makes decision-making difficult and potentially less satisfying. Players struggle to discover games aligning with their specific preferences amidst the noise. Without effective guidance, finding the next engaging experience becomes tedious, reliant on serendipity or extensive manual searching (Ikram & Farooq, 2022). This "discovery problem" necessitates intelligent systems to filter and personalize the vast catalog.

Recommendation systems, powered by Artificial Intelligence (AI) and machine learning, address this challenge. They analyze user behavior patterns and item characteristics to provide personalized suggestions, enhancing user experience by surfacing relevant content (Ricci et al., 2011). For gaming, these systems can significantly improve player satisfaction by connecting users with games they are likely to enjoy but might not have found otherwise, considering unique player motivations and game attributes (Lee & Jung, 2019).

**1.2 Objectives/Aims**

This project aims to address the game discovery problem on platforms like Steam by developing and evaluating different recommendation system approaches. The specific objectives are:

1.  **Develop and Implement Three Recommendation Models:** Implement three distinct recommendation algorithms:
    *   Content-Based Filtering (CBF) using game metadata (titles, tags).
    *   Item-Based Collaborative Filtering (CF) using user-item interaction data (specifically, scaled playtime for positive recommendations).
    *   An Item-Item Similarity model derived from binary positive user recommendations (representing the Matrix Factorization component's implementation focus).
2.  **Generate Personalized Game Recommendations:** Utilize the implemented models to suggest a personalized list of Top-N relevant games based on user input (games they like).
3.  **Develop a Simple User Interface:** Create a basic web application (using Streamlit) to allow users to interact with the different recommendation models and receive game suggestions.
4.  **Evaluate Model Performance:** Assess the effectiveness of the implemented recommendation strategies using appropriate offline evaluation metrics, specifically Precision@K and Recall@K (with K=20), based on a train-test split of historical user interaction data.

**1.3 Motivation**

Developing an effective game recommendation system offers significant value to multiple stakeholders:

*   **Players:** Benefit from personalized discovery, saving time searching and reducing choice fatigue. They are more likely to find games that match their niche interests and playing style, enhancing their overall gaming satisfaction and engagement.
*   **Gaming Platforms (e.g., Steam):** Personalized recommendations drive user engagement, increase time spent on the platform, and boost sales. By surfacing relevant titles, platforms can increase game purchases, DLC sales, and potentially in-game transactions. Companies excelling at personalization drive significantly more revenue (McKinsey, n.d.).
*   **Game Developers:** Gain increased visibility for their games, especially smaller or indie titles that might otherwise be lost in the vast catalog. Reaching the right audience through recommendations can lead to higher sales/downloads and provide valuable feedback loops on player preferences, informing future game development (Hannula et al., 2019).

This project is motivated by the potential to create a more engaging and efficient ecosystem for players, platforms, and developers by leveraging AI to navigate the ever-expanding world of digital games.

---

**Section 2: Research Background**

**2.1 Background of the Applications**

Recommender systems have become ubiquitous tools for navigating information overload in the digital age. Originating from information filtering research (Belkin & Croft, 1992), their core function is to predict user preference and proactively suggest relevant items, thereby reducing cognitive load and enhancing user satisfaction. They are critical components in diverse domains like e-commerce (Amazon), video streaming (Netflix, YouTube), music (Spotify), and social media (Facebook, Twitter).

The fundamental challenge these systems address is connecting users with items from overwhelmingly large catalogs. In the context of digital game distribution platforms like Steam, where tens of thousands of titles exist, this is particularly acute. The gaming domain introduces specific complexities compared to recommending books or movies:
*   **High Engagement Cost:** Games often require significant time investment, making a poor recommendation more impactful.
*   **Diverse Motivations:** Players engage for various reasons – competition, exploration, narrative, social interaction, relaxation, etc. (Lee & Jung, 2019).
*   **Complex Feature Sets:** Game attributes include intricate mechanics, art styles, narrative depth, technical requirements, multiplayer aspects, community integration, and more, which are challenging to capture fully in metadata.
*   **Dynamic Nature:** Games evolve through updates, DLCs, and changing player communities, affecting their relevance over time.

Common approaches to building recommender systems include:

*   **Content-Based Filtering (CBF):** Recommends items similar to those the user liked previously, based on item attributes. It matches user profile features (derived from liked items) with item features. *Strength:* Good for niche tastes, doesn't need other users' data, can recommend new items. *Weakness:* Relies heavily on quality feature extraction, prone to over-specialization (filter bubble), struggles with serendipity.
*   **Collaborative Filtering (CF):** Recommends items based on the preferences of similar users (User-Based CF) or by finding items similar to those the user liked, based on co-occurrence patterns in user interactions (Item-Based CF). It leverages the "wisdom of the crowd." *Strength:* Can find serendipitous recommendations outside a user's typical content profile, doesn't require explicit item features initially. *Weakness:* Suffers from the "cold-start" problem (new users/items), data sparsity can hinder performance, scalability challenges with very large datasets.
*   **Matrix Factorization (MF):** A popular CF technique that decomposes the sparse user-item interaction matrix into lower-dimensional latent factor matrices for users and items. Techniques like Singular Value Decomposition (SVD) or Alternating Least Squares (ALS) are used. It captures hidden correlations. *Strength:* Handles sparsity better than basic CF, captures latent features, often provides accurate predictions. *Weakness:* Can still suffer from cold-start, interpretability of latent factors can be difficult.
*   **Hybrid Approaches:** Combine multiple techniques (e.g., CF and CBF) to leverage their respective strengths and mitigate weaknesses.

This project implements standalone versions of CBF and two variations of Item-Based CF (one using scaled playtime, one using binary interactions, the latter representing the "MF" approach in the context of the provided code's execution).

**2.2 Analysis of Selected Tool with Other Relevant Tools**

The primary development environment chosen for this project was Visual Studio Code (VS Code). This choice was made after considering alternatives like traditional Jupyter Notebook environments and Google Colaboratory (Colab).

| Feature              | Visual Studio Code (Selected)                       | Jupyter Notebook / JupyterLab                       | Google Colaboratory                              |
| :------------------- | :-------------------------------------------------- | :-------------------------------------------------- | :----------------------------------------------- |
| **License**          | MIT License (Code - OSS), Microsoft Dist. Free      | Modified BSD License (Open Source)                  | Proprietary Service (Google)                     |
| **Cost**             | Free                                                | Free (Server costs may apply if self-hosted)        | Freemium (Free tier with limits, Paid Pro/Pro+)  |
| **Year Founded**     | 2015                                                | 2014 (Notebook)                                     | 2017                                             |
| **Founding**         | Microsoft                                           | Project Jupyter (Non-profit, Community)             | Google                                           |
| **Key Features**     | Rich Code Editing, IntelliSense, Debugging, Git Integration, Extensions, Terminal, Notebook Support (via Extension), Remote Dev | Web-based, Cell Execution, Mix Code/Text/Viz, Multi-language Kernels, Sharing | Hosted Jupyter, Free GPU/TPU (limits), Pre-installed Libraries, Drive Integration, Collaboration, Version History |
| **Common Apps**      | General Dev, Web Dev, Data Science (Scripting/Notebooks), Cloud/DevOps | Data Exploration, ML Prototyping, Scientific Computing, Education | Deep Learning, Data Analysis Prototyping, Collaborative ML, Education |
| **Support**          | Large Community, GitHub Issues, Stack Overflow, Docs | Community, GitHub, Mailing Lists, Stack Overflow, Forums | Stack Overflow, Limited Free Support, Paid Google Cloud Support |
| **Limitations**      | RAM usage with many extensions, Notebooks via extension (less native), Requires local setup | Non-linear execution challenges, Version control for `.ipynb` difficult, Less ideal for large software engineering | Resource limits (RAM, Disk, Timeouts) on free tier, Requires internet, Potential privacy concerns, Less suited for large projects |

**2.3 Justify Why the Selected Tool is Suitable**

Visual Studio Code was selected as the most suitable Integrated Development Environment (IDE) for this project due to several key advantages aligning with the project's requirements:

1.  **Streamlined Team Collaboration and Version Control:** As a team project involving multiple members working on different recommendation algorithms, robust version control was paramount. VS Code's native, deeply integrated Git support significantly simplifies collaborative workflows. Features like easy branch management, visual diffing, committing, pushing, and pulling directly within the IDE are more seamless compared to managing `.ipynb` files in standard Jupyter environments. Jupyter notebooks, with their JSON structure containing outputs, often cause merge conflicts in Git, hindering parallel development. While Colab offers Git integration, its nature as a hosted environment with resource limits can complicate workflows, whereas VS Code provides a stable local environment conducive to standard Git practices.
2.  **Unified Interface for AI Development:** VS Code excels at providing a cohesive environment for managing diverse file types common in AI projects. It allows developers to seamlessly work with Python scripts (`.py` for utility functions, preprocessing, model classes), Jupyter notebooks (`.ipynb` for experimentation, visualization, iterative model training – via the powerful Jupyter extension), and data files (`.csv`, etc.) within the same application window, often side-by-side. This integrated experience is more fluid than switching between a file browser and notebook interface in standard Jupyter or managing script imports in Colab, enhancing productivity, especially when structuring code modularly while retaining interactive exploration capabilities.
3.  **Cost-Effectiveness and Local Control:** VS Code is free and open-source (Code - OSS base), eliminating licensing costs for both academic use and potential future development. This contrasts with Colab, where free-tier limitations on resources (GPU/TPU time, RAM, disk space, runtime limits) can quickly become bottlenecks, potentially necessitating paid subscriptions for non-trivial datasets or model training. Furthermore, VS Code operates locally, granting full control over the environment, dependencies, and project files without cloud-imposed restrictions or potential privacy concerns associated with uploading data to third-party servers. Managing numerous experimental notebooks and project versions is straightforward on the local filesystem with Git, unlike the potentially cumbersome versioning within Colab's interface or Drive structure, especially for extensive experimentation.

While Jupyter and Colab offer excellent environments, particularly for pure exploration and leveraging free cloud compute (Colab), VS Code provided the best balance of robust coding features, seamless version control, integrated notebook support, and local control for this collaborative AI development project.

---

**Section 3: Methodology**

**3.1 System Flowchart/Activity Diagram**

*(Conceptual Description, as a visual diagram cannot be embedded here)*

The overall system follows a standard workflow for developing and evaluating recommendation systems:

1.  **Data Loading & Preprocessing:** Start by loading raw data (`recommendations.csv`, `games_tagged.csv`). Filter interactions (e.g., keep only positive recommendations). Process features (e.g., parse tags).
2.  **Exploratory Data Analysis (EDA) & Filtering:** Analyze interaction distributions (user/item counts). Apply filtering thresholds based on EDA to mitigate sparsity and noise (remove users/items with very few interactions). Shuffle the filtered data.
3.  **Model Building (Parallel Paths for each model):**
    *   **CF / MF (Item-Item Cosine):** Construct User-Item Interaction Matrix (sparse). Calculate Item-Item Cosine Similarity Matrix.
    *   **CBF:** Prepare Content Features (TF-IDF on titles/tags). Fit Nearest Neighbors model on TF-IDF matrix.
4.  **Save Artifacts:** Store necessary components for each model (similarity matrices, KNN model, TF-IDF objects, mappings, filtered metadata) using Pickle.
5.  **Evaluation:**
    *   **Train-Test Split:** Split the *filtered interaction data* per user.
    *   **Generate Recommendations:** For each user in the test set, generate Top-K recommendations using their *training data* and the specific model.
    *   **Calculate Metrics:** Compare recommendations against the user's *test set* items using Precision@K and Recall@K.
    *   **Aggregate Results:** Average metrics across all evaluated users.
6.  **User Interface (Streamlit App):**
    *   Load saved model artifacts.
    *   Allow user to select a model (CF, CBF, MF).
    *   Allow user to select game(s) they like (from the games included in the chosen model).
    *   Call the appropriate recommendation function using the selected games and loaded artifacts.
    *   Display the Top-N recommended games with relevant scores.

**3.2 Description of Dataset**

The primary dataset utilized for this project is "Game Recommendations on Steam," publicly available on Kaggle (Kozyriev, n.d.). This dataset originates from Steam, a major digital distribution platform for PC games. It contains millions of user reviews and interactions, making it suitable for building and evaluating recommendation systems in the gaming domain. The key files used are:

*   **`recommendations.csv`**: The core interaction dataset. Contains over 41 million records initially.
    *   `user_id`: Unique identifier for the user.
    *   `app_id`: Unique identifier for the game (item).
    *   `is_recommended_binary`: Binary flag (1 if the user recommended the game in their review, 0 otherwise). **Crucial for filtering positive interactions.**
    *   `hours`: Playtime recorded by the user for the game (used to derive `hours_log_scaled`).
    *   `hours_log_scaled` (from Colab notebooks): Log-transformed and standardized playtime, used as interaction strength in the first CF model.
    *   `review_id`, `helpful`, `funny`, `date`: Metadata about the review itself.
*   **`games_tagged.csv`**: Contains metadata for the games. Approximately 50,872 games initially.
    *   `app_id`: Unique identifier for the game (links to `recommendations.csv`).
    *   `title`: The title of the game. **Used for display and CBF.**
    *   `tags`: A string representation of a list of descriptive tags associated with the game (e.g., 'Action', 'RPG', 'Indie'). **Parsed and used for CBF.**
    *   `positive_ratio`, `user_reviews`: Information about game ratings and review counts.
    *   `price_final`, `price_original`, `discount`: Pricing information.
    *   `win_binary`, `mac_binary`, `linux_binary`, `steam_deck_binary`: Platform compatibility flags.
    *   Other derived features related to price and release date.
*   **`users.csv`**: Provides basic aggregated information about users.
    *   `user_id`: Unique identifier for the user.
    *   `products`: Number of products owned by the user.
    *   `reviews`: Number of reviews written by the user. (Used less directly in the core model building).

This dataset provides rich interaction data (`recommendations.csv`) essential for collaborative approaches and detailed item metadata (`games_tagged.csv`) required for content-based filtering. The filtering steps described in the algorithm sections significantly reduce the size to focus on users and items with sufficient positive interaction history.

**3.3 Description of Algorithm(s)**

**3.3.1 Collaborative Filtering (Item-Based using Scaled Playtime)**

*   **Concept:** Recommends games based on similarity to other games the user has positively interacted with, where similarity is determined by patterns of co-interaction across all users, weighted by engagement (playtime).
*   **Implementation Steps:**
    1.  **Load & Filter (Cell 1):** Load `recommendations.csv`, `games_tagged.csv`. Filter interactions to keep only `is_recommended_binary == 1`.
    2.  **EDA & Thresholding (Cells 2, 2b, 3):** Calculate positive interactions per user/item. Analyze distributions (Figures 3.3.1, 3.3.2 show long tails). Set thresholds (`MIN_USER_INTERACTIONS = 9`, `MIN_ITEM_INTERACTIONS = 2476` based on ~95th percentiles) to address sparsity. Filter interaction data based on these thresholds.
    3.  **Shuffle & Metadata Filter (Cells 4, 5):** Shuffle filtered interactions (`recommendations_pd_filtered`). Filter `games_pd` to match remaining `app_id`s (1872 games).
    4.  **User-Item Matrix (Cell 6):** Create a pivot table (`user_id` x `app_id`) using `hours_log_scaled` as the value for existing positive interactions, `fill_value=0` for others. Convert to CSR sparse matrix (651948 users x 1872 items, ~99.24% sparsity). Create `item_map` and `item_map_inv`. The `hours_log_scaled` value captures interaction intensity.
    5.  **Item-Item Similarity (Cell 7):** Transpose the user-item matrix (`.T`). Calculate pairwise cosine similarity between item rows using `sklearn.metrics.pairwise.cosine_similarity`. Result is a dense 1872x1872 `item_similarity_matrix` where value (i, j) is the similarity between game i and game j based on shared user engagement patterns.
    6.  **Recommendation Function (Cell 8):** `recommend_similar_games_cosine` function defined. Takes `target_app_id`, finds its index using `item_map`, retrieves the corresponding row from `item_similarity_matrix`, sorts scores, excludes self, returns top N `app_id`s (using `item_map_inv`) and titles.
    7.  **Save Artifacts (Cell 9):** Save `item_similarity_matrix`, `item_map`, `item_map_inv`, filtered `games_pd` using `pickle`.
    8.  **Evaluation (Cells 10-13):** Split interactions per user (80% train, 20% test). Define `recommend_for_user` function to aggregate similarity scores from a user's *training* items to rank *other* items. Calculate Precision@20 and Recall@20 against the *test* set for valid users (~348k).

**3.3.2 Content-Based Filtering (CBF using TF-IDF and KNN)**

*   **Concept:** Recommends games based on similarity of their content features (title, tags) to the features of games the user has liked.
*   **Implementation Steps:**
    1.  **Load & Preprocess (Cell 1):** Load `recommendations.csv`, `games_tagged.csv`. Filter for positive interactions (`is_recommended_binary == 1`). Parse `tags` string into lists using `ast.literal_eval`. Create `tags_str` column.
    2.  **EDA & Filtering (Cells 2, 2b, 3):** Perform identical interaction counting and filtering using the same thresholds (`MIN_USER_INTERACTIONS = 9`, `MIN_ITEM_INTERACTIONS = 2476`) on positive interactions.
    3.  **Shuffle & Metadata Filter (Cells 4, 5):** Shuffle filtered positive interactions (`recommendations_pd_filtered`). Filter `games_pd` to match the final 1872 `app_id`s. This filtered `games_pd` is the basis for CBF features.
    4.  **TF-IDF Matrix (Cell 6):** Combine `title` and `tags_str` into a `content` column. Initialize `TfidfVectorizer` (stop words, ngrams 1-2, min_df 2). `fit_transform` on `content` to create `tfidf_matrix` (1872 games x 12051 features, ~99.58% sparse). Each row is a game's content vector. Create `item_map`, `item_map_inv`.
    5.  **KNN Model Fitting (Cell 7):** Initialize `NearestNeighbors` (`metric='cosine'`, `algorithm='brute'`, `n_neighbors=11`). `fit` the model on `tfidf_matrix`. This model efficiently finds similar vectors later.
    6.  **Recommendation Function (Cell 8):** `recommend_similar_games_content` defined. Takes `target_app_id`, finds its TF-IDF vector index using `item_map`. Uses `nn_model.kneighbors()` to find nearest neighbors (indices and distances) in the TF-IDF space. Excludes self, maps indices back to `app_id`s using `item_map_inv`, returns top N games and distances.
    7.  **Save Artifacts (Cell 9):** Save `nn_model`, `tfidf_vectorizer`, `tfidf_matrix`, `item_map`, `item_map_inv`, filtered `games_pd` using `pickle`.
    8.  **Evaluation (Cells 10-13):** Use identical train-test split as CF. Define `recommend_for_user_content_vectorized` which calculates a user's average TF-IDF profile vector from their *training* items. Performs a *single* KNN search using this profile vector against all items. Filters out training items, returns top K neighbors. Calculate Precision@20 and Recall@20 against the test set for a *sample* of 10,000 valid users.

**3.3.3 Matrix Factorization (Implemented as Item-Item Cosine Similarity on Binary Interactions)**

*   **Concept:** Recommends games based on similarity derived from co-occurrence in positive user recommendations. While named MF, the code implements Item-Item CF using binary interactions.
*   **Implementation Steps:**
    1.  **Load & Filter (Cell 1):** Load `recommendations.csv`, `games_tagged.csv`. Filter interactions for `is_recommended_binary == 1`.
    2.  **EDA & Thresholding (Cells 2, 3, 4):** Identical interaction counting and filtering using thresholds (`MIN_USER_INTERACTIONS = 9`, `MIN_ITEM_INTERACTIONS = 2476`) on positive interactions.
    3.  **Shuffle & Metadata Filter (Cells 5, 6):** Shuffle filtered positive interactions (`df_final`). Filter `games` metadata to match the final 1872 `app_id`s.
    4.  **User-Item Matrix (Cell 7):** Create pivot table (`user_id` x `app_id`) using `is_recommended_binary` (value '1') for positive interactions, `fill_value=0`. Convert to CSR sparse matrix (shape and sparsity identical to first CF model). Create `user_map`, `item_map`, and inverses. Here, '1' simply indicates a positive recommendation occurred.
    5.  **Item-Item Similarity (Cell 8):** Transpose user-item matrix. Calculate pairwise cosine similarity using `sklearn.metrics.pairwise.cosine_similarity`. Store result in dense `item_similarity_matrix` (1872x1872). Similarity now based purely on overlap of users giving positive recommendations.
    6.  **Recommendation Function (Cell 9):** `recommend_similar_games_cosine` function defined (identical logic to first CF model), using the binary-interaction-based similarity matrix.
    7.  **Save Artifacts (Cell 10):** Save `item_similarity_matrix`, `item_map`, `item_map_inv`, filtered `games` DataFrame using `pickle` (with `_cosine` suffix).
    8.  **Evaluation (Cells 11-14):** Use identical train-test split. Use the same `recommend_for_user_evaluation` function (as first CF model) which aggregates similarity scores from training items. Calculate Precision@20 and Recall@20 against the test set for *all* valid users (~503k).
    9.  **Top-N CSV (Cell 15):** Additional step to pre-calculate and save top 10 similar games for every game based on this binary interaction similarity matrix (`top10_similar_games_cosine.csv`).

**3.4 Proposed Test Plan/Hypothesis**

The primary goal of the testing phase is to evaluate the effectiveness of the three implemented recommendation algorithms (Content-Based Filtering, Item-Item Collaborative Filtering with scaled playtime, Item-Item Collaborative Filtering with binary interactions) in predicting user preferences for games.

**Hypothesis:** All three recommendation models, trained on the filtered Steam interaction and metadata, will be able to recommend relevant games to users with reasonable accuracy, as measured by Precision@20 and Recall@20 metrics on a held-out test set of user interactions. We expect variations in performance between the models, potentially with the Collaborative Filtering approaches (which leverage user behavior patterns) achieving higher recall than the Content-Based approach (which relies solely on available metadata).

**Test Plan:**
1.  Split the filtered positive interaction dataset into training (80%) and testing (20%) sets, stratified by user.
2.  For each model:
    *   Build/Train the model using only the training data (CF/MF: calculate similarity matrix; CBF: build TF-IDF/fit KNN).
    *   For each user designated for evaluation (those with sufficient training data and items in the test set):
        *   Generate the top 20 game recommendations based on their training interactions using the specific model's logic.
        *   Ensure recommendations do not include items already present in the user's training set.
    *   Compare the generated Top-20 recommendations against the items the user actually interacted with positively in the test set.
    *   Calculate Precision@20 and Recall@20 for this user.
3.  Aggregate the metrics by calculating the average Precision@20 and Recall@20 across all evaluated users for each model.
4.  Compare the average metric scores across the three models.

---

**Section 4: Result**

**4.1 Results**

The three recommendation models were evaluated using the plan outlined above. The key performance metrics obtained were:

| Model                                                      | Evaluated Users | Average Precision@20 | Average Recall@20 | Notes                                        |
| :--------------------------------------------------------- | :-------------- | :------------------- | :---------------- | :------------------------------------------- |
| **Collaborative Filtering (Playtime Weighted)**            | ~348,481        | 0.0355               | 0.1903            | Based on `hours_log_scaled` interactions   |
| **Content-Based Filtering (TF-IDF + KNN)**                 | 10,000 (Sample) | 0.0178               | 0.0984            | Based on Title + Tags; Evaluated on sample |
| **Matrix Factorization (Binary Interaction Similarity)** | ~502,739        | 0.0372               | 0.2359            | Based on `is_recommended_binary=1`         |

*Note: The slight difference in the number of evaluated users between the two CF/MF approaches might stem from minor variations in how valid users were filtered or handled during the split process across potentially different runs or notebook states, despite aiming for identical logic.*

**User Interface (Streamlit Application)**

A web application was developed using Streamlit (`app.py`) to provide an interactive interface for the recommendation models.

*   **(Conceptual Screenshot 1: Main Interface)** The UI presents radio buttons allowing the user to select one of the three models: "Content-Based Filtering", "Collaborative Filtering", or "Matrix Factorization". Below this, a multiselect dropdown allows the user to search and select one or more games they like from a list relevant to the chosen model. A slider lets the user adjust the desired number of recommendations (Top-N, default 10).
*   **(Conceptual Screenshot 2: CF/MF Recommendations)** When the "Collaborative Filtering" or "Matrix Factorization" model is selected and games like 'Counter-Strike: Global Offensive' are chosen, the application displays a table of recommended games. The table includes columns for Rank ('No.'), 'App ID', 'Recommended Game' title, and an 'Interaction Score'. Games frequently played by the same users (e.g., other Valve titles, popular FPS games) appear with high scores.
*   **(Conceptual Screenshot 3: CBF Recommendations)** When "Content-Based Filtering" is selected and a game like 'Terraria' is chosen, the recommendation table shows games with similar tags (e.g., 'Sandbox', 'Survival', '2D', 'Crafting') like 'Starbound' or 'Don't Starve'. The score column is labeled 'Content Score', representing content similarity (1 - cosine distance).

The application successfully loads the saved artifacts for the selected model and calls the corresponding recommendation function, displaying the results in a user-friendly table.

**4.2 Discussion/Interpretation**

The evaluation results provide insights into the relative performance of the different recommendation strategies on this dataset:

1.  **Collaborative vs. Content-Based:** Both Collaborative Filtering approaches (Playtime Weighted and Binary Interaction) significantly outperformed the Content-Based Filtering model in terms of both Precision@20 and Recall@20.
    *   **MF (Binary Interaction):** Precision@20=0.0372, Recall@20=0.2359
    *   **CF (Playtime Weighted):** Precision@20=0.0355, Recall@20=0.1903
    *   **CBF (TF-IDF/KNN):** Precision@20=0.0178, Recall@20=0.0984 (on a smaller sample)
    This suggests that, for this dataset and task, user behavior patterns (captured by CF/MF) are stronger predictors of future positive interactions than the available content features (title, tags used by CBF). The "wisdom of the crowd" appears more effective here than relying solely on metadata. The tags, while useful, might not capture the nuanced gameplay elements or player experiences that drive co-interaction patterns.

2.  **Binary Interaction vs. Playtime Weighted CF:** The Item-Item similarity model based on binary positive interactions slightly outperformed the one using scaled log-playtime, particularly in Recall@20 (0.2359 vs. 0.1903). This is somewhat counter-intuitive, as one might expect playtime to be a stronger signal. Possible interpretations include:
    *   The simple act of recommending (`is_recommended_binary=1`) might be a cleaner signal of preference than the potentially noisy `hours` data (users might leave games running, playtime scaling might not perfectly capture engagement intensity).
    *   The binary approach might capture broader co-occurrence patterns more effectively.

3.  **Magnitude of Metrics:** The absolute values for Precision@20 (around 0.03-0.04 for CF/MF) might seem low. However, in recommendation system evaluations, especially with large catalogs and implicit feedback characteristics, such values are common. It means that out of 20 recommendations, typically less than one (on average) was found in the user's held-out positive interactions. Despite this, the recommendations can still be valuable for *discovery*. Recall@20 values (around 0.19-0.24 for CF/MF) indicate that the models recommended roughly 19-24% of the relevant items (from the test set) within the top 20 suggestions.

4.  **CBF Performance:** The lower performance of CBF highlights the potential limitations of relying solely on title and tags. Richer content features (e.g., detailed descriptions, user-generated reviews text analysis, more granular genre/mechanic classifications) might be needed to improve its effectiveness. The evaluation was also on a smaller sample, though the trend seems clear.

5.  **UI Observations:** The Streamlit application demonstrates the practical utility of the models. The recommendations generated *qualitatively* appear reasonable within the logic of each model (CF/MF suggesting behaviorally related games, CBF suggesting content-similar games), providing distinct types of suggestions that could be useful to different users or in different contexts (e.g., finding something *exactly* like a game vs. finding what *players* of that game also play).

In summary, the results indicate that collaborative filtering methods based on user interactions were more effective than the implemented content-based approach for this dataset. The simple binary indication of a positive recommendation proved slightly more effective for recall than using scaled playtime as the interaction signal in the item-item similarity calculation.

---

**Section 5: Discussion and Conclusion**

**5.1 Achievements**

This project successfully achieved its primary objectives:

1.  **Implemented Three Recommendation Models:** Three distinct recommendation algorithms were successfully implemented using Python and relevant libraries (Pandas, Scikit-learn, Scipy): Content-Based Filtering (TF-IDF/KNN), Item-Item Collaborative Filtering (Cosine Similarity on scaled playtime), and another Item-Item Collaborative Filtering variant (Cosine Similarity on binary positive interactions).
2.  **Generated Personalized Recommendations:** Functions were developed for each model capable of generating Top-N game recommendations based on a set of input games liked by the user.
3.  **Developed User Interface:** A functional Streamlit application was created, allowing users to select a model, input preferred games, and receive tailored recommendations interactively.
4.  **Evaluated Model Performance:** All models were evaluated using offline metrics (Precision@20, Recall@20) on a held-out test set, providing quantitative insights into their effectiveness and allowing for comparative analysis. The results demonstrated measurable differences in performance between the approaches.

The project successfully navigated the process from data loading and preprocessing, through model building and evaluation, to deploying a simple interactive prototype, fulfilling the core requirements of the assignment.

**5.2 Limitations and Future Works**

Despite the achievements, the project has several limitations, opening avenues for future improvement:

1.  **Data Limitations:**
    *   **Positive Interactions Only:** The models primarily relied on explicit positive recommendations (`is_recommended_binary=1`). Negative signals or implicit interactions (e.g., playtime without recommendation, wishlist additions, purchases) were not fully leveraged, which could provide a more holistic view of user preference.
    *   **Data Sparsity:** Although filtering was applied, sparsity remains inherent. This particularly affects users/items near the filtering threshold and makes recommendations for niche tastes difficult.
    *   **Cold-Start Problem:** None of the implemented models explicitly address the cold-start problem for new users (no interaction history) or new items (no interactions yet). CBF can handle new items if their content features are available, but not new users. CF/MF struggle with both.
2.  **Algorithmic Limitations:**
    *   **CBF Feature Richness:** The CBF model relied only on titles and tags. Incorporating more features like game descriptions, user review text, developer/publisher information, or more granular genre/mechanic classifications could significantly improve its performance.
    *   **Model Complexity:** The project focused on classic, relatively simpler models (Item-Item CF, KNN). More advanced techniques were not explored:
        *   **True Matrix Factorization:** Implementing MF using algorithms like ALS or SVD could capture latent factors more effectively than the binary interaction similarity approach.
        *   **Deep Learning:** Neural network-based models (e.g., Two-Tower models, Neural Collaborative Filtering) can capture complex non-linear relationships and easily incorporate diverse features.
        *   **Hybrid Models:** Combining CBF and CF approaches could leverage the strengths of both, potentially improving accuracy and handling cold-start better.
3.  **Evaluation Limitations:**
    *   **Offline Metrics:** Evaluation relied solely on offline metrics (Precision/Recall). These don't always perfectly correlate with online user satisfaction or business goals (e.g., engagement, diversity, serendipity).
    *   **Static Split:** The train-test split was static. Time-aware evaluation (predicting future interactions based on past data) would be more realistic.
    *   **Limited Metrics:** Other metrics like NDCG (considers ranking position), coverage, diversity, and serendipity were not measured.

**Future Works:**

*   Incorporate implicit feedback signals (playtime, purchases, wishlists) and negative feedback.
*   Explore advanced Matrix Factorization techniques (ALS, SVD) and Deep Learning models.
*   Develop hybrid recommendation strategies combining content and collaborative signals.
*   Implement strategies to mitigate the cold-start problem (e.g., using content features for new items, asking new users for initial preferences).
*   Enhance CBF feature engineering using NLP on descriptions/reviews or incorporating image features.
*   Conduct online A/B testing to evaluate models based on real user interactions and business metrics.
*   Expand evaluation to include metrics like NDCG, diversity, and serendipity.
*   Implement user-based collaborative filtering as an alternative or complement to item-based approaches.

**Conclusion:**

This project successfully demonstrated the implementation and evaluation of Content-Based and Collaborative Filtering recommendation systems for Steam games. Collaborative approaches based on user interaction patterns showed superior performance compared to the content-based model using titles and tags, highlighting the power of leveraging community behavior. The developed Streamlit application provides a practical interface for exploring these different recommendation paradigms. While limitations exist, particularly concerning data richness, algorithm complexity, and evaluation scope, the project provides a solid foundation and identifies clear directions for future work in building more sophisticated and effective game recommendation systems.

---

**References**

Belkin, N. J., & Croft, W. B. (1992). Information filtering and information retrieval: Two sides of the same coin? *Communications of the ACM*, *35*(12), 29–38. https://doi.org/10.1145/138859.138861

Dar, A., & Gul, M. (2024). The “less is better” paradox and consumer behaviour: a systematic review of choice overload and its marketing implications. *Qualitative Market Research an International Journal*, *28*(1), 122-145. https://doi.org/10.1108/qmr-01-2024-0006

Hannula, R., Nikkilä, A., & Stefanidis, K. (2019). Gamerecs: video games group recommendations. In *ECIR 2019: Advances in Information Retrieval* (pp. 513–524). Springer International Publishing. https://doi.org/10.1007/978-3-030-30278-8_49

Ikram, F., & Farooq, H. (2022). Multimedia recommendation system for video game based on high-level visual semantic features. *Scientific Programming*, *2022*, 1–12. https://doi.org/10.1155/2022/6084363

Kozyriev, A. (n.d.). *Game Recommendations on Steam*. Kaggle. Retrieved from https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

Lee, Y., & Jung, Y. (2019). A mapping approach to identify player types for game recommendations. *Information*, *10*(12), 379. https://doi.org/10.3390/info10120379

McKinsey & Company. (n.d.). *The value of getting personalization right–or wrong–is multiplying*. Retrieved from https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/the-value-of-getting-personalization-right-or-wrong-is-multiplying

Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2011). *Recommender Systems Handbook*. Springer US. https://doi.org/10.1007/978-0-387-85820-3 (Note: While the provided link was to ResearchGate, the book is a standard reference)

Rypdal, K. (2018). Empirical growth models for the renewable energy sector. *Advances in Geosciences*, *45*, 35–44. https://doi.org/10.5194/adgeo-45-35-2018

SteamSpy. (2024). *Games released per year*. Retrieved from https://steamspy.com/year/ (Note: Access date should be added if possible, data retrieved approx. mid-2024 based on report context)

**Tools Used:**
*   **Programming Language:** Python 3.x
*   **Core Libraries:** Pandas, NumPy, Scikit-learn, SciPy, Matplotlib
*   **Development Environment:** Visual Studio Code
*   **Web Framework:** Streamlit
*   **Serialization:** Pickle

---