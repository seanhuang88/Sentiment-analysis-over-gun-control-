#------------------------------------------------
# Prepare for classification using Decision Tree
#------------------------------------------------

#Install necessary library
install.packages("gsub")
install.packages("rtweet")
install.packages("tm")
install.packages("NLP")
install.packages("SnowballC")
install.packages("slam")
install.packages("wordcloud")
install.packages("gmodels")

# FOr KNN Classification
install.packages("class")

# FOr Decision Tree Classification
install.packages("party")

# FOr Naive Bayes Classification
install.packages("e1071")


library("gsub")
library("rtweet")
library("tm")
library("NLP")
library("SnowballC")
library("slam")
library("wordcloud")
library("class")
library("gmodels")
library("party")
library("e1071")


# search for 1000 tweets with hashtag "guncontrol" in USA, excluding retweet.
rt <- search_tweets("#guncontrol", geocode = lookup_coords("usa") , include_rts = FALSE, n=1000)

# Saves as flattened CSV file of Twitter data
write_as_csv(rt, "MyData.tweets", prepend_ids = TRUE, na = "", fileEncoding = "UTF-8")

# Reads Twitter data that was previously saved as a CSV file.
dataset <- read_twitter_csv("MyData.tweets.csv")

# Clean up character vector (tweets) to more of a plain text.
datasetPlain <- plain_tweets(dataset)

# Clean up duplication.
sapply(datasetPlain, function(x) length(unique(x)))

# Get the date of each tweet
datasetPlain$created_at <- strtrim(datasetPlain$created_at,10)

# Concatenate same people's tweet in same day, assuming one will stick to his or her opinion in a day. Get rid of similar tweets from the same person.
datasetCon <- aggregate(datasetPlain$text, by=list(datasetPlain$user_id,datasetPlain$created_at), paste)

#set column name to "c1","c2","text"
dataset <- setNames(datasetCon, c("c1","c2","text"))

# Cleaning hashtag and mention
dataset$text = gsub("[@#][a-z,A-Z,[:digit:]]*", "", dataset$text)

# cleaning UTF8
dataset$text = gsub("[<].*[>]", "", dataset$text)

# trim space
dataset$text <- trimws(dataset$text)

# trim c("
dataset$text <- gsub('c("', "",dataset$text, fixed = TRUE)

# cleaning tweet without words
dataset <- dataset[!dataset$text=="", ]

# Saves as flattened CSV file of Twitter data
write_as_csv(dataset, "MyData.csv", prepend_ids = TRUE, na = "", fileEncoding = "UTF-8")

# Manually label 200 tweets and divide dataset into dataset with for class & against class

datasetFor <- read_twitter_csv("MyDataWithForClass.csv")

# Create a vector source
my.docs <- VectorSource(datasetFor$text)

# Representing and computing on corpora
my.corpus <- Corpus(my.docs)

# Remove punctuation marks from a text document
my.corpus <- tm_map(my.corpus, removePunctuation)

# Stem words in a text document using Porter's stemming algorithm
my.corpus <- tm_map(my.corpus, stemDocument)

# Remove numbers from a text document
my.corpus <- tm_map(my.corpus, removeNumbers)

# Convert letter to lower case
my.corpus <- tm_map(my.corpus, content_transformer(tolower))

# Strip extra whitespace from a text document
my.corpus <- tm_map(my.corpus, stripWhitespace)

# Remove various kinds of stopwords for English
my.corpus <- tm_map(my.corpus, removeWords, c('gun',stopwords("english")))

# Plot a word cloud
wordcloud(my.corpus, max.words = 100, random.order = FALSE)

# do the same for Against class
datasetAnainst <- read_twitter_csv("MyDataWithAgainstClass.csv")

my.docs <- VectorSource(datasetAnainst$text)
# ... Follow same steps as from line 70 to 91



#------------------------------------------------
# Prepare for classification using KNN
#------------------------------------------------

datasetClass <- read_twitter_csv("MyDataClassified.csv")

my.docs <- VectorSource(datasetClass$text)
# ... Follow same steps as from line 88 to 106

# Constructs or coerces to a term-document matrix or a document-term matrix
term.doc.matrix.stm <- TermDocumentMatrix(my.corpus)

# Remove sparse terms from a document-term or term-document matrix
term.doc.matrix.stm.trim <- removeSparseTerms(term.doc.matrix.stm, sparse=0.99)

term.doc.matrix <- as.matrix(term.doc.matrix.stm.trim)

# Returns the transpose
doc.term.matrix <- t(term.doc.matrix)

# Split the iris dataset into training and test set
train_index <- sample(1:nrow(doc.term.matrix), 0.75 * nrow(doc.term.matrix))

train.set <- doc.term.matrix[train_index,]

test.set  <- doc.term.matrix[-train_index,]

# Store the labels from our training and test datasets
train_labels <- datasetClass$For.Class[train_index]

test_labels <- datasetClass$For.Class[-train_index]

# Prediction on the test set
knn_prediction <- knn(train = train.set, test = test.set, cl=train_labels, k = 2)

# Confusion matrix
CrossTable(x=test_labels, y=knn_prediction, prop.chisq=TRUE)
conf.mat <- table("Predictions" = knn_prediction, Actual = test_labels)

# Accuracy
accuracy <- sum(diag(conf.mat))/length(test.set) * 100
 # Result : [1] 0.1484099



#------------------------------------------------
# Prepare for classification using Decision Tree
#------------------------------------------------

# continus use doc.term.matrix
train.set <- doc.term.matrix[train_index,]

# attach class label to dataset
labels <- datasetClass$For.Class[train_index]
train.set <- cbind(train.set, labels)

# convert to dataframe
train.set <- as.data.frame(train.set)

# Apply conditional inference trees and run model on the training set
tweet_ctree <- ctree(labels ~ ., data=train.set)

# same steps for test.set 
# (I have problem when I first try, error message is " Levels in factors of new data do not match original data". 
# Need wrote the final preprocessed file into a csv, and read it again to a dataframe,then applied those test data on the model)
# Reason behind : Because there were few categorical value in test dataframe, which even after removal, 
# was there in the list with 0 rows(which doesn't occur in training dataset).

test.set  <- doc.term.matrix[-train_index,]
labels <- datasetClass$For.Class[-train_index]
test.set <- cbind(test.set, labels)
test.set <- as.data.frame(test.set)

# prediction on the test set
ctree_prediction <- predict(tweet_ctree, test.set)

# confusion matrix
conf.mat <- table(ctree_prediction, labels)

# Accuracy
accuracy <- sum(diag(conf.mat))/length(test.set) * 100
# Result [1] : 0.2261484



#------------------------------------------------
# Prepare for classification using Naive Bayes
#------------------------------------------------

# Function to convert the word frequencies to yes (presence) and no (absence) labels, because for sentiment classification, word occurrence matters more than word frequency
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

datasetClass <- read_twitter_csv("MyDataClassified.csv")

datasetClass$For.Class <- as.factor(datasetClass$For.Class)

my.docs <- VectorSource(datasetClass$text)
# ... Follow same steps as from line 88 to 106

# Use 'findFreqTerms' function to indentify the frequent words,  
# then restrict the DTM to use only the frequent words using the 'dictionary' option
dtm <- DocumentTermMatrix(my.corpus)

fivefreq <- findFreqTerms(dtm, 5)

doc.term.matrix <- DocumentTermMatrix(my.corpus, control=list(dictionary = fivefreq))

# Split the iris dataset into training and test set
train_index <- sample(1:nrow(doc.term.matrix), 0.75 * nrow(doc.term.matrix))
doc.term.matrix <- apply(doc.term.matrix, 2, convert_count)
train.set <- doc.term.matrix[train_index,]
test.set  <- doc.term.matrix[-train_index,]
train_labels <- datasetClass$For.Class[train_index]

# Train the classifier
classifier <- naiveBayes(train.set, train_labels, laplace = 1)

# Testing the Predictions
pred <- predict(classifier, newdata=test.set)
# I get error "Error in `[.default`(object$tables[[v]], , nd + islogical[attribs[v]]) : subscript out of bounds"
