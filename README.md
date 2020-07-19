
#LGG Implementation
This is an implementation ofConcept learning algorith Least General Generalization using Conjunction of common literals algorithm 4.1 & internal disjunction of conjunctions algorithm 4.2. Refer Peter Flach.Machine Learning: The Art and Science of Algorithms ThatMake Sense of Data.  Cambridge University Press, USA, 2012 for more information.

##Required libraries
Pandas & Sklearn

##Introduction
The aim of the assignment is to implement a concept learner and to verify that it works as expected using the features and instances given in the spambase dataset \cite{Dua:2019}. The document discusses the implementation of the concept learner and the results obtained from it.

##Dataset
The dataset used for this purpose is the spambase dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Spambase). The spambase dataset consists of 57 features and 4601 rows . The data contains continuous values for the 57 features. There are 1812 spam positives and 2789 spam negatives in the data. The spam positives constitute approximately 39% of the data.

##Data Preparation
Data has been extracted using pandas library and loaded into a pandas dataframe. There are three integer values and remaining are real values in the dataset. Out of which, one is output of the dataset which is an integer value. To do Least General Generalization (LGG), the values should be converted into descrete integer vallues. So, a uniform discretization transform (Equal width binning or descetization) of the dataset was performed on the real values in the dataset. uniform discretization transform will preserve the probability distribution of each input variable but will make it discrete with the specified number of ordinal groups or labels. The number of bins or ordinal groups assigned were 10, thereby giving us 10 values for all the 55 continuous real values in the dataset. 

##Methodology
The number of possible instances with the given feature values were obtained by multiplying all the unique values in each feature. For the first 55 features since binning was applied with 10 bins, it is 10^55, for the remaining 2 input features, the number of unique values are 271 and 919. So, the total possible instances are 271 * 919  * 10^55 = 2.49049 * 10^60.

Both algorithms 4.2 and 4.3 were implemented using algorithm 4.1. Algorithm 4.2 uses conjunction of common literals to obtain the least general generalization of the concept. While implementing, the hypothesis space is initiated with the values of first feature and is taken as list of 57 feature lists and iterated on the remaining data. All the literals common are collected in the list assigned for each feature, thereby satisfying conjunction of conjunctions. An empty list in the hypothesis space says that it is a general feature and any value could be there to fit into the hypothesis space. When trained on 80% of positives in the data, the size of the hypothesis space for the concept learnt is 2.49049 x 10^49. When trained on 100% of positives in the data, the size of the hypothesis space is 2.49049 * 10^52.
 \newline
 
 Algorithm 4.3 uses internal disjunction of conjunctions to obtain the least general generalization of the concept. While implementing, the hypothesis space is initiated with the values of first feature and is taken as list of 57 feature lists and iterated on the remaining data. The features values which weren't in the hypothesis space were added into it, thereby fulfilling internal disjunction of conjunctions. If an instance has all feature values which are in the respective feature lists in the hypothesis space, then it fits into the hypothesis space. When trained on 80\% of positives in the data, the size of the hypothesis space for the concept learnt is 1.177 x 10^34. When trained on 100\% of positives in the data, the size of the hypothesis space is 6.045 x 10^36.
 
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
Input : conjunctions x , y . 
Output : conjunction z.
1 z ←true;
2 for each feature f do
3 	if f = vx is a conjunct in x and f = vy is a conjunct in y then
4 		add f = Combine-ID(vx,vy) to z; //Combine-ID:seetext
5 		end 
6 	end
7 return z

##Results
When algorithm 4.1 and 4.2 trained on 80\% of the positives and tested on the test data containing 20\% of the positives and all the negatives, it resulted in accuracy of 36.14%. And when tested on 100\% positives, it resulted in an accuracy of 0.9983. Hence, it isn't a complete hypothesis as expected. And when tested on all data, it resulted in an accuracy of 56.27%.

When algorithm 4.1 and 4.2 trained on 100\% of the positives and tested on the test  100\% positives, it resulted in an accuracy of 1.0. Hence, it is a complete hypothesis as expected. And when tested on all of the data, it resulted in an accuracy of 53.14%.

There wasn't any significant change in accuracy or hypothesis space when trained on 100\% positives and 80\% data. It was also observed the hypothesis space remains the same even when the training data is shrinked to 10\%.

When algorithm 4.1 and 4.3 trained on 80% of the positives and tested on the test data containing 20\% of the positives and all the negatives, it resulted in accuracy of 56.14%. And when tested on 100\% positives, it resulted in an accuracy of 94.31%. Hence, it is not a complete hypothesis as expected. And when tested on all data, it resulted in an accuracy of 69.94%.

When algorithm 4.1 and 4.3 trained on 100\% of the positives and tested on the test  100\% positives, it resulted in an accuracy of 100\%. Hence, it is a complete hypothesis as expected. And when tested on all of the data, it resulted in an accuracy of 67.98%. Though the hypothesis space is complete it resulted in a lower accuracy on the data.

All the hypothesis spaces obtained are inconsistent as they can not identify negative cases with a 100\% accuracy.

