from nltk.corpus import movie_reviews as reviews
from nltk.classify import NaiveBayesClassifier as classify
from nltk.classify.util import accuracy as classifier_accuracy

def code():
    possitive_fileid = reviews.fileids('pos')#pos/cv000 to cv999
    negative_fileid = reviews.fileids('neg')#neg/cv000 to cv999
   
    print ("Total number of reviews in the dataset: "+str(len(reviews.fileids())))
    print ("Total number of possitive Reviews: "+str(len(possitive_fileid)))
    print ("Total number of negative Reviews: "+str(len(negative_fileid)))

    possitive_features = [(extract(reviews.words(fileids=[fd])),'Good') for fd in possitive_fileid]
    negative_features = [(extract(reviews.words(fileids=[fd])),'Bad') for fd in negative_fileid]
   
    train_features = possitive_features[:] + negative_features[:]
    test_features = possitive_features[:] + negative_features[:]  

    print('\nTotal number of trained datapoints:', len(train_features))
    print('Total number of tested datapoints:', len(test_features))

   # Train a Naive Bayes classifier
    classifier = classify.train(train_features)
    acc = classifier_accuracy(classifier, test_features)*100
    print('\nAccuracy of the system: '+str(acc)+' %')

    n = 20
    print('\nTop ' + str(n) + ' most informative words:')
    for i, item in enumerate(classifier.most_informative_features()):
        print(str(i+1) + '. ' + item[0])
        if i == n - 1:
            break
   
    default_reviews=[
        'The costumes in this movie were great',
        'I think the story was terrible and the characters were very weak',
        'People say that the director of the movie is amazing',
        'This is such an idiotic movie, i will not recommend it to anyone',
        'This is not the movie i recommend']
   
    print("\nCurrent Reviews:\n")
    for i in default_reviews:
        print(i+'.\n\n')
   
    while (True):
        a = int(input("Do you want to add more reviews, if yes press 1 else press 0\n"))
        if(a == 0):
            break
        elif(a == 1):
            b = input("Please enter the review\n")
            default_reviews.append(b)

    total_rating = 0
   
    print("\nMovie Review Predictions:")
    for review in default_reviews:
        print("\nReview:", review)

        probabilities = classifier.prob_classify(extract(review.split()))

        predicted_sentiment = probabilities.max()

        print("Predicted sentiment:", predicted_sentiment)
        print("Probability of correct sentiment:", format(round(probabilities.prob(predicted_sentiment),2), '.2f'))

        if(predicted_sentiment == 'Good'):
            total_rating += round(probabilities.prob(predicted_sentiment), 2)
        else:
            if predicted_sentiment == 'Bad' :
                total_rating -= round(probabilities.prob(predicted_sentiment), 2)

   
    if(-0.25 <= total_rating <= 0.25):
        print("\n\nOverall Rating: Average")
    elif total_rating < -0.25:
        print("\n\nOverall Rating: Very Bad")
    else:
        print("\n\nOverall Rating: Very good")
       
def extract (words):
    return dict([(w, True) for w in words])
           
def val():
    inp = int(input("Please press 1 to enter the Movie Rating System else press 2 to exit\n"))
   
    if (inp == 1):
        code()
    elif (inp == 2):
        print("Have a nice day\n")
        exit
    else:
        print("Invalid input, please try again")
        val()
   
if __name__ == '__main__':
    val()