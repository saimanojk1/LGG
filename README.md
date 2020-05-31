
#LGG Implementation
This is an implementation ofConcept learning algorith Least General Generalization using Conjunction of common literals algorithm 4.1 & internal disjunction of conjunctions algorithm 4.2. Refer Peter Flach.Machine Learning: The Art and Science of Algorithms ThatMake Sense of Data.  Cambridge University Press, USA, 2012 for more information.

##Required libraries
Pandas & Sklearn

##Introduction
The aim of the assignment is to implement a concept learner and to verify that it works as expected using the features and instances given in the spambase dataset \cite{Dua:2019}. The document discusses the implementation of the concept learner and the results obtained from it.

##Dataset
The dataset used for this purpose is the spambase dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Spambase). The spambase dataset consists of 57 features and 4601 rows . The data contains continuous values for the 57 features. There are 1812 spam positives and 2789 spam negatives in the data. The spam positives constitute approximately 39\% of the data.

##Data Preparation
Data has been extracted using pandas library and loaded into a pandas dataframe. Since the data values are percentage of words present in 100 words of a text, the data has been standardized and all the values  within the standard deviation are set to zero to rule out all the instances which have normal amount of each feature. It was observed that there are  no features in the  data below standard deviation. All the features above standard deviation were rounded off to the nearest integer making the data discrete completely. Rounding off them to nearest integers instead of making their value '1' is to obtain different levels of presence of a feature, there by increasing the number of values for each feature. This helps us in identify anomalous behaviour better. The training for hypothesis is done completely on positive instances of the data.

##Methodology
The number of possible instances with the given feature values were obtained by multiplying all the unique values in each feature plus one (for absence of any value for the feature), and were obtained to be 8.92 x 10^65.
Both algorithms 4.2 and 4.3 were implemented using algorithm 4.1. Algorithm 4.2 uses conjunction of common literals to obtain the least general generalization of the concept. While implementing, the hypothesis space is initiated with the values of first feature and is taken as list of 57 feature lists and iterated on the remaining data. All the literals common are collected in the list assigned for each feature, thereby satisfying conjunction of conjunctions. An empty list in the hypothesis space says that it is a general feature and any value could be there to fit into the hypothesis space. When trained on 80\% of positives in the data, the size of the hypothesis space for the concept learnt is 2.91 x 10^59. When trained on 100\% of positives in the data, the size of the hypothesis space is 2.91 x 10^65.
 \newline
 
 Algorithm 4.3 uses internal disjunction of conjunctions to obtain the least general generalization of the concept. While implementing, the hypothesis space is initiated with the values of first feature and is taken as list of 57 feature lists and iterated on the remaining data. The features values which weren't in the hypothesis space were added into it, thereby fulfilling internal disjunction of conjunctions. If an instance has all feature values which are in the respective feature lists in the hypothesis space, then it fits into the hypothesis space. When trained on 80\% of positives in the data, the size of the hypothesis space for the concept learnt is 1.36 x 10^48. When trained on 100\% of positives in the data, the size of the hypothesis space is 7.04 x 10^49.
 
##Algorithms

###Algorithm 4.1: LGG-Set(D ) – find least general generalisation of a set of instances.

Input : data D.
Output : logical expression H . x ←first instance from D ; H←x;
while instances left do
x ←next instance from D;
H ←LGG(H,x) ; end
return H
// e.g., LGG-Conj (Alg. 4.2) or LGG-Conj-ID (Alg. 4.3)

###Algorithm 4.2: LGG-Conj(x,y) – find least general conjunctive generalisation of two conjunctions.

Input : conjunctions x , y .
Output : conjunction z.
1 z ←conjunction of all literals common to x and y;
2 return z

###Algorithm 4.3: LGG-Conj-ID(x,y) – find least general conjunctive generalisation of two conjunctions, employing internal disjunction.
  1 2 3 4 5 6 7
Input : conjunctions x , y . Output : conjunction z.
z ←true;
for each feature f do
iff =vx isaconjunctinxandf =vy isaconjunctinythen
addf =Combine-ID(vx,vy)toz; //Combine-ID:seetext
end end
return z

##Results
When algorithm 4.1 and 4.2 trained on 80\% of the positives and tested on the test data containing 20\% of the positives and all the negatives, it resulted in accuracy of . And when tested on 100\% positives, it resulted in an accuracy of 1.0. Hence, it is a complete hypothesis as expected. And when tested on all data, it resulted in an accuracy of 51.72\%.

When algorithm 4.1 and 4.2 trained on 100\% of the positives and tested on the test  100\% positives, it resulted in an accuracy of 1.0. Hence, it is a complete hypothesis as expected. And when on tested on all of the data, it resulted in an accuracy of 51.72\%.

There wasn't any significant change in accuracy or hypothesis space when trained on 100\% positives and 80\% data. It was also observed the hypothesis space remains the same even when the  training data is shrinked to 10\%.
\newline

When algorithm 4.1 and 4.3 trained on 80\% of the positives and tested on the test data containing 20\% of the positives and all the negatives, it resulted in accuracy of . And when tested on 100\% positives, it resulted in an accuracy of 98.56\%. Hence, it is not a complete hypothesis as expected. And when tested on all data, it resulted in an accuracy of 68.87\%.

When algorithm 4.1 and 4.3 trained on 100\% of the positives and tested on the test  100\% positives, it resulted in an accuracy of 100\%. Hence, it is a complete hypothesis as expected. And when on tested on all of the data, it resulted in an accuracy of 67.39\%. Though the hypothesis space is complete it resulted in a lower accuracy on the data.

All the hypothesis spaces obtained are inconsistent as they can not  identify negative cases with a 100\% accuracy.

