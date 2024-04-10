# Data Structures
import numpy  as np
import pandas as pd

# Corpus Processing
import re
import nltk.corpus
from unidecode                        import unidecode
from nltk.tokenize                    import word_tokenize
from nltk                             import SnowballStemmer
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.preprocessing            import normalize
from sklearn.decomposition import PCA

# K-Means
from sklearn.cluster import KMeans

# Visualization and Analysis
import matplotlib.pyplot  as plt


def career_func(desired_career):
    data = pd.read_csv('final_grades.csv')

    course = data.pop('course')

    course_list = course.values.tolist()

    course_list_new =[]
    course_index_list = []
    course_code_list = []
    for i in course_list:
        try:
            if i!='subject subjec':
                course_code_list.append(i.split('-')[0])
            else:
                data.drop(course_list.index(i),axis=0,inplace=True)
                data.reset_index(drop=True,inplace=True)
            if i.split('-')[1]!='':
                course_list_new.append(i.split('-')[1:])
            else:
                course_list_new.append('')
                continue
        except:
            continue
        
    course_code = pd.DataFrame(course_code_list)
    course_code.columns=['Course Code']

    course_list_new_1 = []
    for i in course_list_new:
        j = ' '.join(i)
        course_list_new_1.append(j)
    course_new = pd.DataFrame(course_list_new_1)
    course_new.columns = ['Course Names']
    course_list_new_1 = list(set(course_list_new_1))
        
    # with open('subjectsNamesInFinalGrades.txt', 'w') as f:
    #     for line in course_list_new_1:
    #         f.write(f"{line}\n")
            
    course = pd.concat([course_code,course_new],axis=1)

    final_grades = pd.concat([course,data],axis=1)

    # with open('subjectsNamesInFinalGrades.txt', 'r') as f:
    corpus = course_list_new_1.copy()

    k=0
    for i in corpus:
        corpus[k] = i.strip('\n')
        k = k+1

    # removes a list of words (ie. stopwords) from a tokenized list.
    def removeWords(listOfTokens, listOfWords):
        return [token for token in listOfTokens if token not in listOfWords]

    # applies stemming to a list of tokenized words
    def applyStemming(listOfTokens, stemmer):
        return [stemmer.stem(token) for token in listOfTokens]

    # removes any words composed of less than 2 or more than 21 letters
    def twoLetters(listOfTokens):
        twoLetterWord = []
        for token in listOfTokens:
            if len(token) <= 2 or len(token) >= 21:
                twoLetterWord.append(token)
        return twoLetterWord

    nltk.download('stopwords')
    nltk.download('punkt')

    def processCorpus(corpus, language):   
        stopwords = nltk.corpus.stopwords.words(language)
        param_stemmer = SnowballStemmer(language)
        
        for document in corpus:
            index = corpus.index(document)
            corpus[index] = corpus[index].replace(u'\ufffd', '8')   # Replaces the ASCII '�' symbol with '8'
            corpus[index] = corpus[index].replace(',', '')          # Removes commas
            corpus[index] = corpus[index].rstrip('\n')              # Removes line breaks
            corpus[index] = corpus[index].casefold()                # Makes all letters lowercase
            
            corpus[index] = re.sub('\W_',' ', corpus[index])        # removes specials characters and leaves only words
            corpus[index] = re.sub("\S*\d\S*"," ", corpus[index])   # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
            corpus[index] = re.sub("\S*@\S*\s?"," ", corpus[index]) # removes emails and mentions (words with @)
            corpus[index] = re.sub(r'http\S+', '', corpus[index])   # removes URLs with http
            corpus[index] = re.sub(r'www\S+', '', corpus[index])    # removes URLs with www

            listOfTokens = word_tokenize(corpus[index])
            twoLetterWord = twoLetters(listOfTokens)

            listOfTokens = removeWords(listOfTokens, stopwords)
            listOfTokens = removeWords(listOfTokens, twoLetterWord)
            
            # listOfTokens = applyStemming(listOfTokens, param_stemmer)

            corpus[index]   = " ".join(listOfTokens)
            corpus[index] = unidecode(corpus[index])

        return corpus

    language = 'english'
    corpus = processCorpus(corpus, language)
    corpus

    vectorizer = TfidfVectorizer()

    # TD-IDF Matrix
    X = vectorizer.fit_transform(corpus)

    # extracting feature names
    tfidf_tokens = vectorizer.get_feature_names_out()

    # reduce the dimensionality of the data using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X.toarray())

    tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names_out())

    final_df = tf_idf

    print("{} rows".format(final_df.shape[0]))
    final_df.T.nlargest(5, 0)




    def run_KMeans(max_k, data):
        max_k += 1
        kmeans_results = dict()
        for k in range(2 , max_k):
            kmeans = KMeans(n_clusters = k
                            , init = 'k-means++'
                            , n_init = 10
                            , random_state = 1)

            kmeans_results.update( {k : kmeans.fit(data)} )
            
        return kmeans_results



    # Running Kmeans
    k = 8
    kmeans_results = run_KMeans(k, final_df)




    def get_top_features_cluster(tf_idf_array, prediction, n_feats):
        labels = np.unique(prediction)
        dfs = []
        for label in labels:
            id_temp = np.where(prediction==label) # indices for each cluster
            x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
            sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
            features = vectorizer.get_feature_names_out()
            best_features = [(features[i], x_means[i]) for i in sorted_means]
            df = pd.DataFrame(best_features, columns = ['features', 'score'])
            dfs.append(df)
        return dfs

    # def plotWords(dfs, n_feats):
    #     plt.figure(figsize=(8, 4))
    #     for i in range(0, len(dfs)):
    #         plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
    #         sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
    #         plt.show()
            
            
            
    best_result = 5
    kmeans = kmeans_results.get(best_result)

    final_df_array = final_df.to_numpy()
    prediction = kmeans.predict(final_df)
    n_feats = 20
    dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
    # plotWords(dfs, 13)

    # # plot the results
    # colors = ['red', 'green', 'blue', 'yellow', 'gray']
    cluster = ['Mechanical Engineering','Engineering Design Process','Environmental Engineering','Physics and Engineering','Finance']
    # for i in range(5):
    #     plt.scatter(reduced_data[kmeans.labels_ == i, 0],
    #                 reduced_data[kmeans.labels_ == i, 1],
    #                 s=10, color=colors[i],
    #                 label=f' {cluster[i]}')
    # plt.legend()
    # plt.show()


    profile_list = []
    for i in prediction:
        profile_list.append(cluster[i])
    cluster_names = pd.DataFrame(profile_list)

    cluster_names.columns = ['Profile Names']

    # with open('subjectsNamesInFinalGrades.txt', 'r') as f:
    subject_names = course_list_new_1.copy()

    k=0
    for i in subject_names:
        subject_names[k] = i.strip('\n')
        k = k+1
        
    subject_names = pd.DataFrame(subject_names)
    subject_names.columns = ['Subject Names']

    subject_names_with_clusters = pd.concat([subject_names, cluster_names],axis=1)

    subject_names_with_clusters.rename(columns = {'0':'Profile Names'}, inplace = True)

    subject_names_with_clusters.set_index('Subject Names')

    list_of_subjects = course_list_new_1

    final_grades.insert(2,"Profile", None, True)

    result = final_grades.sort_values(by=['Course Names','session']).drop_duplicates(subset='Course Names', keep='first')

    result.set_index('Course Names',inplace=True)

    for i in list_of_subjects:
        if i in result.index:
            result.loc[i,'Profile'] = subject_names_with_clusters['Profile Names'][list_of_subjects.index(i)]
            # print('Yes')

    result = result.sort_values(by='average', ascending=False)

    result.replace('#DIV/0!', np.nan, inplace=True)

    result.dropna(inplace=True)
    
    result = result.reset_index()

    # result.to_csv('final_grades along with profile.csv')

    final_result = result[result['Profile'] == desired_career]

    return final_result
