from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
#from nltk.tokenize import sent_tokenize -- nije uspeo adekvatno da podeli recenice
import numpy as np
import networkx as nx


def read_text(file_name):
    file = open(file_name,"r", encoding="utf8")
    text = file.read()
    file.close()

    # svi karakteri koji nisu u opsegu su zamenjeni whitespace-om
    text = text.replace("[^a-zA-Z]", " ")

    # ideja ja da recenice mogu biti odvojene enterom, a i tackom, pa moramo da ih splitujemo i po enteru
    # i po tacki
    sentences = text.split('\n')
    newSentences = []
    # ponovo splitujemo po ". " i appendujemo u novu listu
    for sentence in sentences:
        if ". " in sentence:
            s = sentence.split(". ")
            newSentences += s
        else:
            newSentences.append(sentence)
    del sentences

    # brisemo clanove liste koji su prazni stringovi
    for sentence in newSentences:
        if sentence == '':
            newSentences.remove(sentence)

    sentences = []
    # sada svaku recenicu treba pretvoriti u listu reci da bi svaka rec posle bila vektor, a ne svako slovo
    for sentence in newSentences:
        sentences.append(sentence.split(" "))

    return sentences


def similarity_matrix(sentences, stop_words):
    sim_matrix = np.zeros((len(sentences),len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_matrix[i,j] = cosine_sent_similarity(sentences[i], sentences[j], stop_words)

    return sim_matrix



def sent_to_vec(s1,s2):
    all_words = list(set(s1 + s2))
    vector1 = np.zeros(len(all_words))
    vector2 = np.zeros(len(all_words))
    return all_words, vector1, vector2


def cosine_sent_similarity(s1, s2, stop_words=None):
    if stop_words is None:
        stop_words = []

    # konvertujemo u lowercase
    s1 = [w.lower() for w in s1]
    s2 = [w.lower() for w in s2]

    # treba nam vektor svih reci i vektor za svaku recenicu
    all_words, vector1, vector2 = sent_to_vec(s1,s2)

    for w in s1:
        if w not in stop_words:
            vector1[all_words.index(w)] += 1

    for w in s2:
        if w not in stop_words:
            vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1,vector2)



if __name__ == '__main__':
    sentences = read_text('test.txt')
    stop_words = ["a", "ako", "ali", "bi", "bih", "bila", "bili", "bilo", "bio", "bismo", "biste", "biti", "bumo", "da",
                  "do", "duž", "ga", "hoće", "hoćemo", "hoćete", "hoćeš", "hoću", "i", "iako", "ih", "ili", "iz", "ja",
                  "je", "jedna", "jedne", "jedno", "jer", "jesam", "jesi", "jesmo", "jest", "jeste", "jesu", "jim",
                  "joj", "još", "ju", "kada", "kako", "kao", "koja", "koje", "koji", "kojima", "koju", "kroz", "li",
                  "me", "mene", "meni", "mi", "mimo", "moj", "moja", "moje", "mu", "na", "nad", "nakon", "nam", "nama",
                  "nas", "naš", "naša", "naše", "našeg", "ne", "nego", "neka", "neki", "nekog", "neku", "nema", "netko",
                  "neće", "nećemo", "nećete", "nećeš", "neću", "nešto", "ni", "nije", "nikoga", "nikoje", "nikoju",
                  "nisam", "nisi", "nismo", "niste", "nisu", "njega", "njegov", "njegova", "njegovo", "njemu", "njezin",
                  "njezina", "njezino", "njih", "njihov", "njihova", "njihovo", "njim", "njima", "njoj", "nju", "no",
                  "o", "od", "odmah", "on", "ona", "oni", "ono", "ova", "pa", "pak", "po", "pod", "pored", "prije", "s",
                  "sa", "sam", "samo", "se", "sebe", "sebi", "si", "smo", "ste", "su", "sve", "svi", "svog", "svoj",
                  "svoja", "svoje", "svom", "ta", "tada", "taj", "tako", "te", "tebe", "tebi", "ti", "to", "toj",
                  "tome", "tu", "tvoj", "tvoja", "tvoje", "u", "uz", "vam", "vama", "vas", "vaš", "vaša", "vaše", "već",
                  "vi", "vrlo", "za", "zar", "će", "ćemo", "ćete", "ćeš", "ću", "što", "takođe", "takodje"]

    #stop_words_eng = stopwords.words('english')


    # Napravimo matricu slicnosti
    sim_matrix = similarity_matrix(sentences, stop_words)

    # Matricu konvertujemo u graf
    sim_graph = nx.from_numpy_array(sim_matrix)

    # PageRank algoritam rangira recenice
    scores = nx.pagerank(sim_graph)

    # Sortiranje po rangu
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summarized_text = []
    # Biranje prvih n recenica
    n = 10
    for i in range(n):
        summarized_text.append(" ".join(ranked_sentence[i][1]))

    print("Summarize Text: \n", "\n ".join(summarized_text))
