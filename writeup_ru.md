[//]: # (Image References)

[image1]: ./my_images/bar_chart_before.png "Dataset before"
[image2]: ./my_images/bar_chart_after.png "Dataset after"
[image3]: ./my_images/affine_trans.png "affine transforma"
[image4]: ./my_images/perspective_trans.png "perspective transform"
[image5]: ./my_images/to_gray.png "to gray scale"
[image6]: ./my_images/1.png "Traffic Sign 1"
[image7]: ./my_images/2.png "Traffic Sign 2"
[image8]: ./my_images/3.png "Traffic Sign 3"
[image9]: ./my_images/4.png "Traffic Sign 4"
[image10]: ./my_images/5.png "Traffic Sign 5"
[image11]: ./my_images/6.png "Traffic Sign 6"
[image12]: ./my_images/7.png "Traffic Sign 7"
[image13]: ./my_images/8.png "Traffic Sign 8"

### Введение
Данный проект выполнен в рамках Nano degree программы "Self driving car" [Udacity](https://www.udacity.com/drive).
Суть данного проекта: классификация дорожных знаков Германии.

### Немного о данных

Набор данных представляет из себя базу дорожных знаков Германии. Саму базу можно скачать [сдесь]().
Я использовал Pandas и Numpy для получения статистики о данных: 

* Количество тренировочных данных 34799 дорожных знаков
* Количество данных, на которых проводилась валидация, 4410 дорожных знаков
* Количество тестовых данных 12630 дорожных знаков
* Каждый дорожный знак представляет из себя RGB картинку размером 32 x 32 x 3
* Все дорожные знаки сгруппированы в 43 класса

#### Сводная информация о данных.

На русунке ниже представлен BarChart показывающий распределение дорожных знаков в по классам.

![alt text][image1]

Можно видеть, что мы имеем дело с несбалансированным набором данных. Дисбаланс достигает 10 кратной разницы
между минимум (180) и максимум (2010). Не сбалансированные данные - это беда нейронных сетей.
Потому что, при малом количестве примеров сеть не может правильно обобщить, а вместо этого просто запомнит 
запомнить предоставленные примеры. Поэтому, для начала, мы должны расширить наш набор данных для получения 
более сбалансированного набора данных.

### Разработка и тестирование модели

#### Предобработка данных
Для получения данных с которыми может работать наша модель я сделал несколько действий.
Во первых, все картинки были преобразованы к цветовой схеме градаций серого.

![alt text][image5]

Во вторых, набор данных был расширин, добавлением в каждый класс новых, сгенерированные на основе существующих, картинок.
Количество картинок подбиралось таким образом, что бы получить в результате сбалансированный набор данных.
Для генерации использовались несколько техник:
Первое - это афинное преобразование.

![alt text][image3]

Второе - это горизонтальная и вертикальная трансформация перспективы.

![alt text][image4]

Как результат, после всех преобразований, был получен набор данных представленный на рисунке ниже.

![alt text][image2]

Новый набор данных имеет равное количество дорожных знаков для всех классов и составляет 4020 картинок для каждого класса.

#### Финальный вариант архитектуры нейронной сети.

Мой вариант нейронной сети содержит нижеследующие слои:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image                       |
| Convolution 1x1   	| 1x1 stride, valid padding, outputs 32x32x3    |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 30x30x32   |
| ReLU			        |												|
| Max pooling           | 2x2 stride, 
| Max pooling	      	| 2x2 stride, outputs 16x16x64 	  			    |
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


