## Recognition of EOG eye-witten characters

(version 1) paper link : https://www.mdpi.com/2079-9292/10/23/3009

### Data preprocessing

> - Donghyun Kang
> 
> We conducted three pre-processing steps - Eye blink removal, normalization, and data resampling. <br>
> 
> In the version 1 <br>
> Our preprocessing pipeline has only two steps as a data normalizaion and data reampling using dataset after eye blink removing. <br>
> The pipeline constructed the 5 stages with 8 functions following 'Data normalization -> Removing closed points -> Fill points -> Distance re-adjust -> Points Equalization & Minimization -> Distance re-adjust' <br>
>
> In the version 2 <br>
> We refactored these pipeline. <br>
> The new pipeline contains 4 functinos following 'Data normalization -> Removing closed points -> Fill points & Points Minimization -> Distance re-adjust' <br>
> And, we added more datasets for the robust experiments as a 'without eye blink removing characters' and 'katakana characters' <br>
>
> As the result, we could save the time for preprocessing from 40.7s to 15.1s using the 'eye blink removed characters'

![preprocessing](https://user-images.githubusercontent.com/48580174/137443414-b34f5afb-d677-44f3-b6da-d8a98243805a.png)


### Experiments of classification

> In the version 1 <br>
> We selected an algorithm which called 'Siamese Network' to classify the eog characters. <br> 
> To extract the features of the input, we designed a 'base model' that was constructed as a 1D CNN, LSTM and hierarchical attention layer <br> 
> In addition, We used new characters for post-processing and data argumentation which named 'Reference data'. <br>
> These dataset became the benchmark of each class. <br>
> 
> In the version 2 <br>
> We still used 'Siamese Network' as a main algorithm. <br>
> However, Vision Transformer(ViT) was used as a 'base model' and compared to previous base model. <br>
> The new base model was reconstructed for our dataset. <br>
> In the experiment of version 1, we got the problem that the accuracy of classification using the 'non trained class' was low.
> We used the combination of numbers and katakana characters to solve this problem. <br>



